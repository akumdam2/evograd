"""What the coding-agent CLIs (Claude Code, Codex) share: workspace, metered evaluator, deliverable.

Each of ``candidates/claude_code/run.py`` and ``candidates/codex/run.py`` is::

    workspace = Workspace(task, ledger, agent, args)   # TASK.md, candidate.py, evaluate.py
    status = launch(<the CLI, headless, on prompt(task, budget)>, workspace.root, ...)
    best, extra = workspace.collect()                  # the program to measure
    finish(args, task, ledger, best, ...)

Both agents run as shipped, headless, on the same prompt (``prompt``): a short
paragraph on the workspace, then the task's problem statement verbatim (also in
``TASK.md``). The workspace holds:

    TASK.md                the shared problem statement
    candidate.py           the deliverable; starts as the task's skeleton
    evaluate.py            IMMUTABLE. Sends a module to the evaluation server
                           (full by default, --correctness for correctness only)
                           and prints its verdict
    adbench_primitives.py  granted primitives, importable standalone

The workspace is a fresh directory outside the repository (``--workspace``
overrides), so the agent does not start beside other agents' results. It is
copied to ``<run-dir>/workspace`` afterwards.

Nothing the agent is given names the repository, the run or the task pack:
evaluate.py reaches the evaluator through a Linux abstract Unix socket served
by the driver (:class:`Evaluator`), which owns the budget and the ledger and
redacts those paths from the feedback it returns; the agent's environment is
scrubbed of variables that name them (:meth:`Workspace.environment`). The
agent still runs as the same user, so a search of the filesystem or the
process list can find the repository; only a container would close that.

Budget mapping: ``--budget N`` is enforced exactly. The server runs at most N
evaluations of either depth, one at a time (evaluations time on the GPU, so
two must not overlap). The agent may also run its own scripts on the GPU; that
is its own work, as it is for GEAK's engineers.

Deliverable: ``candidate.py`` when the server passed exactly that module (the
agent's own choice); otherwise the best correct module it evaluated (a full
evaluation's ``combined_score`` first, then the latest); otherwise
``candidate.py`` as it is, unless it is still the skeleton.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import socket
import socketserver
import struct
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path

import _common

CANDIDATE = "candidate.py"
ORACLE = "evaluate.py"
WORKSPACE = "workspace"
PROMPT = "prompt.md"


def add_arguments(p) -> None:
    p.add_argument("--workspace", type=Path, default=None,
                   help="empty directory the agent works in (default: a new temporary directory)")
    p.add_argument("--effort", default=None, help="the agent's reasoning effort (default: its own)")
    p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    p.add_argument("--eval-timeout", type=int, default=900, help="seconds per evaluation")
    p.add_argument("--time-limit", type=int, default=6 * 3600, help="seconds for the session")


def cli_version(executable: str) -> str | None:
    """``<executable> --version``, for run.json: the CLIs update themselves."""
    try:
        done = subprocess.run([executable, "--version"], capture_output=True, text=True,
                              timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = [line.strip() for line in done.stdout.splitlines() if line.strip()]
    return lines[-1] if lines else None


def prompt(task, budget: int) -> str:
    abi = task.spec["abi"]
    return f"""You are working alone in a directory set up for one GPU kernel task; no one will \
answer questions. The problem statement is below and in TASK.md.

- Write your module to `{CANDIDATE}`. It starts as an unimplemented skeleton.
- `python {ORACLE}` runs the evaluator that judges this task on `{CANDIDATE}` and prints its \
verdict, the measured latencies and `combined_score` (higher is better). \
`python {ORACLE} --correctness` checks correctness only, without timing. \
`python {ORACLE} PATH` evaluates another module.
- `{ORACLE}` runs at most {budget} evaluations in total (both kinds count) and prints how many \
remain after each. After that it refuses.
- You may run your own scripts, tests and profilers on the GPU. Work inside this directory and \
do not modify `{ORACLE}`.
- When you stop, `{CANDIDATE}` must hold your best module defining `{abi['forward_fn']}` and \
`{abi['backward_fn']}`: it is what gets measured. If its final contents never passed \
`{ORACLE}`, the best correct module you evaluated is measured instead.
- Keep improving `combined_score` until you are satisfied or the evaluations are used up, then \
stop.

---

{task.statement}"""


def oracle_source(address: str, budget: int) -> str:
    return f'''#!/usr/bin/env python3
"""The evaluator for this task (generated by evograd adbench-tools). Do not edit.

    python evaluate.py                     # candidate.py: correctness, timing, combined_score
    python evaluate.py --correctness       # correctness only, no timing
    python evaluate.py [--correctness] PATH

The verdict and the score come from the evograd evaluator, the one that
measures every agent on this task. At most {budget} evaluations of either kind,
one at a time; each call prints how many remain. Exit 0 iff the module is
correct, 3 once the evaluations are used up.
"""
import json
import socket
import sys
from pathlib import Path

ADDRESS = {address!r}

args = sys.argv[1:]
if "-h" in args or "--help" in args:
    print(__doc__)
    sys.exit(0)
correctness = "--correctness" in args
paths = [a for a in args if a != "--correctness"]
candidate = Path(paths[0] if paths else "candidate.py")
if not candidate.is_file():
    print(f"FAIL -- no module at {{candidate}}")
    sys.exit(1)

request = {{"source": candidate.read_text(encoding="utf-8"), "correctness": correctness}}
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
    try:
        connection.connect(ADDRESS)
    except OSError as exc:
        print(f"FAIL -- the evaluator is not reachable ({{exc}})")
        sys.exit(1)
    connection.sendall((json.dumps(request) + "\\n").encode())
    reply = json.loads(connection.makefile(encoding="utf-8").readline())
print(reply["text"])
sys.exit(reply["exit"])
'''


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        size = struct.calcsize("3i")
        _, uid, _ = struct.unpack("3i", self.request.getsockopt(socket.SOL_SOCKET,
                                                                socket.SO_PEERCRED, size))
        if uid != os.getuid():
            return
        try:
            reply = self.server.respond(json.loads(self.rfile.readline()))
        except Exception as exc:  # the agent sees the harness failing, not a hang
            reply = {"exit": 1, "text": f"FAIL -- the evaluator failed: {type(exc).__name__}: {exc}"}
        reply["text"] = self.server.redact(reply["text"])
        self.wfile.write((json.dumps(reply) + "\n").encode())


class Evaluator(socketserver.UnixStreamServer):
    """The service behind evaluate.py: the budget, the ledger, redacted feedback.

    One request at a time, so evaluations never overlap on the GPU. The address
    is in Linux's abstract namespace (no file), and only this user may connect.
    """

    request_queue_size = 64

    def __init__(self, task, ledger, origin: str, budget: int, timeout: int):
        self.task, self.ledger, self.origin = task, ledger, origin
        self.budget, self.timeout = budget, timeout
        self.address = f"\0adbench-eval-{uuid.uuid4().hex}"
        super().__init__(self.address, _Handler)
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    def used(self) -> int:
        from evograd.adbench_tools.ledger import EVALUATIONS_FILE, read_lines

        return sum(1 for e in read_lines(self.ledger.root / EVALUATIONS_FILE)
                   if e.get("origin") == self.origin)

    def respond(self, request: dict) -> dict:
        from evograd.adbench_tools.evaluate import evaluate_source

        if self.used() >= self.budget:
            return {"exit": 3, "text": f"BUDGET EXHAUSTED: all {self.budget} evaluations are "
                                       "used. Leave your best module in candidate.py and stop."}
        record = evaluate_source(self.task, request["source"], run_dir=self.ledger.root,
                                 origin=self.origin, timeout=self.timeout,
                                 correctness_only=bool(request.get("correctness")))
        verdict = "CORRECT" if record.correct else "INCORRECT"
        return {"exit": 0 if record.correct else 1,
                "text": f"{record.feedback()}\n{verdict}\n"
                        f"evaluations remaining: {self.budget - self.used()} of {self.budget}"}

    def redact(self, text: str) -> str:
        """Replace the run, task pack and repository paths with labels, longest first."""
        labels = {str(self.ledger.root): "<run>", str(self.task.root): "<task>",
                  str(_common.REPO): "<evograd>"}
        for path in sorted(labels, key=len, reverse=True):
            text = text.replace(path, labels[path])
        return text

    def stop(self) -> None:
        self.shutdown()
        self.server_close()


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


class Workspace:
    """The directory a coding agent works in, its evaluator, and the program it leaves."""

    def __init__(self, task, ledger, origin: str, args):
        self.task, self.ledger, self.origin = task, ledger, origin
        self.temporary = args.workspace is None
        root = Path(tempfile.mkdtemp(prefix=f"adbench_{origin}_")) if self.temporary \
            else args.workspace
        self.root = root.resolve()
        if self.root.exists() and any(self.root.iterdir()):
            raise SystemExit(f"{self.root} is not empty")
        self.root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(task.root / "TASK.md", self.root / "TASK.md")
        shutil.copy2(task.root / "skeleton.py", self.root / CANDIDATE)
        primitives = task.root / "adbench_primitives.py"
        if primitives.is_file():
            shutil.copy2(primitives, self.root / primitives.name)
        self.evaluator = Evaluator(task, ledger, origin, args.budget, args.eval_timeout)
        (self.root / ORACLE).write_text(oracle_source(self.evaluator.address, args.budget),
                                        encoding="utf-8")
        self.oracle_sha256 = _sha256(self.root / ORACLE)
        (ledger.root / PROMPT).write_text(prompt(task, args.budget), encoding="utf-8")

    def environment(self, gpu: str) -> dict[str, str]:
        """The caller's environment on one GPU, without the paths the agent is not given.

        A variable naming the repository, the run or the task pack loses those
        entries if it is a search path (PATH, PYTHONPATH, ...) and is dropped
        otherwise (PWD, OLDPWD, SLURM_SUBMIT_DIR, ...).
        """
        hidden = (str(_common.REPO), str(self.ledger.root), str(self.task.root))
        env = {}
        for name, value in os.environ.items():
            parts = value.split(os.pathsep)
            kept = [part for part in parts if not any(path in part for path in hidden)]
            if len(kept) == len(parts):
                env[name] = value
            elif kept and len(parts) > 1:
                env[name] = os.pathsep.join(kept)
        env["PWD"] = str(self.root)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
        return env

    def collect(self) -> tuple[Path | None, dict]:
        """Stop the evaluator, copy the workspace into the run, choose the program to measure."""
        self.evaluator.stop()
        kept = self.ledger.root / WORKSPACE
        if kept != self.root:
            shutil.copytree(self.root, kept, symlinks=True, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__"))
            if self.temporary:
                shutil.rmtree(self.root, ignore_errors=True)
        best, deliverable = choose(self.task, self.ledger, self.origin, kept / CANDIDATE)
        return best, {"deliverable": deliverable, "workspace": str(self.root),
                      "oracle_intact": _sha256(kept / ORACLE) == self.oracle_sha256}


def choose(task, ledger, origin: str, final: Path) -> tuple[Path | None, str]:
    """The program to measure, and which rule picked it (see the module docstring)."""
    from evograd.adbench_tools.artifact import package, unpackage
    from evograd.adbench_tools.ledger import EVALUATIONS_FILE, read_lines

    def body(text: str) -> str | None:
        try:
            return unpackage(package(task, text, origin=origin))
        except ValueError:
            return None

    passed = [e for e in read_lines(ledger.root / EVALUATIONS_FILE)
              if e.get("origin") == origin and e.get("correct") and e.get("program")
              and Path(e["program"]).is_file()]
    text = final.read_text(encoding="utf-8") if final.is_file() else None
    if text is not None and text == (task.root / "skeleton.py").read_text(encoding="utf-8"):
        text = None
    mine = body(text) if text is not None else None
    if mine is not None and any(body(Path(e["program"]).read_text(encoding="utf-8")) == mine
                                for e in passed):
        return final, "final"
    if passed:
        def rank(item):
            index, entry = item
            full = entry.get("kind") == "full"
            return full, float(entry.get("combined_score") or 0.0) if full else 0.0, index

        return Path(max(enumerate(passed), key=rank)[1]["program"]), "best_evaluated"
    return (final, "final_unverified") if text is not None else (None, "none")


def launch(command: list[str], cwd: Path, env: dict[str, str], log: Path, time_limit: int,
           input_text: str | None = None) -> str:
    """Run an agent CLI with its stdout in ``log``; the session's status.

    ``input_text`` goes to the CLI's stdin through a pipe. The CLI gets its own
    process group, and the whole group is killed when the session ends, so
    nothing the agent started keeps the GPU busy during the final measurement.
    An interrupt (``run_all.sh``'s time limit) ends the session the same way
    and lets the driver go on to score.
    """
    with log.open("w", encoding="utf-8") as out, \
            log.with_suffix(".stderr.txt").open("w", encoding="utf-8") as err:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=out, stderr=err, start_new_session=True,
            stdin=subprocess.DEVNULL if input_text is None else subprocess.PIPE)
        if input_text is not None:
            try:
                process.stdin.write(input_text.encode())
            except BrokenPipeError:
                pass
            finally:
                process.stdin.close()
        try:
            process.wait(timeout=time_limit)
            status = "completed" if process.returncode == 0 else f"exit_{process.returncode}"
        except subprocess.TimeoutExpired:
            status = "time_limit"
        except KeyboardInterrupt:
            status = "interrupted"
        if process.poll() is None:
            _signal(process, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        _signal(process, signal.SIGKILL)
        process.wait()
    return status


def _signal(process: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(process.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass
