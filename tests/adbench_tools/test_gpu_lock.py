"""CPU subprocess tests for GPU ownership, timeouts, and cancellation."""
import concurrent.futures
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from evograd.adbench_tools.gpu_lock import gpu_lock, run_evaluation


class GPULockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lock = self.root / "gpu.lock"
        self.env = {**os.environ, "ADBENCH_GPU_LOCK": str(self.lock)}

    def test_unset_lock_preserves_output_and_exit_code(self):
        env = dict(self.env)
        env.pop("ADBENCH_GPU_LOCK")
        result = run_evaluation([sys.executable, "-c", "print('ok'); raise SystemExit(3)"],
                                env=env, timeout=2, label="plain")
        self.assertEqual((result.returncode, result.stdout.strip()), (3, "ok"))
        self.assertFalse(self.lock.exists())

    def test_parallel_evaluations_do_not_overlap_or_timeout_while_waiting(self):
        events = self.root / "events.jsonl"
        code = (
            "import os,json,time; "
            "assert 'ADBENCH_GPU_LOCK' not in os.environ; "
            f"f=open({str(events)!r},'a',buffering=1); "
            "f.write(json.dumps(['start',os.getpid()])+'\\n'); "
            "time.sleep(0.7); f.write(json.dumps(['end',os.getpid()])+'\\n')"
        )
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            jobs = [pool.submit(run_evaluation, [sys.executable, "-c", code],
                                env=self.env, timeout=1, label=f"case{i}") for i in range(2)]
            self.assertEqual([job.result().returncode for job in jobs], [0, 0])
        rows = [json.loads(line) for line in events.read_text().splitlines()]
        self.assertEqual([row[0] for row in rows], ["start", "end", "start", "end"])
        self.assertEqual(rows[0][1], rows[1][1])
        self.assertEqual(rows[2][1], rows[3][1])
        log = Path(str(self.lock) + ".log").read_text()
        self.assertEqual(log.count("acquire"), 2)
        self.assertEqual(log.count("release"), 2)
        self.assertGreater(max(map(float, re.findall(r"waited=([0-9.]+)", log))), 0.4)

    def descendant_code(self, marker, started):
        child = f"import time; from pathlib import Path; time.sleep(1); Path({str(marker)!r}).touch()"
        return ("import subprocess,sys,time; from pathlib import Path; "
                f"subprocess.Popen([sys.executable,'-c',{child!r}]); "
                f"Path({str(started)!r}).touch(); time.sleep(10)")

    def test_timeout_kills_descendants_before_releasing_lock(self):
        marker, started = self.root / "escaped", self.root / "started"
        with self.assertRaises(subprocess.TimeoutExpired):
            run_evaluation([sys.executable, "-c", self.descendant_code(marker, started)],
                           env=self.env, timeout=0.3, label="timeout")
        self.assertTrue(started.exists())
        result = run_evaluation([sys.executable, "-c", "print('next')"],
                                env=self.env, timeout=2, label="next")
        self.assertEqual(result.returncode, 0)
        time.sleep(1.1)
        self.assertFalse(marker.exists())

    def test_termination_cleans_gpu_process_group(self):
        marker, started = self.root / "escaped", self.root / "started"
        metadata = self.root / "status.json"
        command = [sys.executable, "-m", "evograd.adbench_tools.gpu_lock", "_worker",
                   "cancel", "20", str(metadata), sys.executable, "-c",
                   self.descendant_code(marker, started)]
        proc = subprocess.Popen(command, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            until = time.monotonic() + 5
            while not started.exists() and time.monotonic() < until:
                time.sleep(0.02)
            self.assertTrue(started.exists())
            proc.send_signal(signal.SIGTERM)
            proc.communicate(timeout=3)
            self.assertEqual(proc.returncode, 143)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        time.sleep(1.1)
        self.assertFalse(marker.exists())
        self.assertIn("release", Path(str(self.lock) + ".log").read_text())

    def test_waiting_monitor_can_be_cancelled_without_acquiring_gpu(self):
        started = self.root / "unexpected-start"
        metadata = self.root / "status.json"
        command = [sys.executable, "-m", "evograd.adbench_tools.gpu_lock", "_worker",
                   "waiting", "2", str(metadata), sys.executable, "-c",
                   f"from pathlib import Path; Path({str(started)!r}).touch()"]
        with patch.dict(os.environ, {"ADBENCH_GPU_LOCK": str(self.lock)}):
            with gpu_lock("holder"):
                proc = subprocess.Popen(command, env=self.env,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    time.sleep(0.2)
                    proc.terminate()
                    proc.communicate(timeout=3)
                    self.assertEqual(proc.returncode, 143)
                    self.assertFalse(started.exists())
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.communicate()


if __name__ == "__main__":
    unittest.main()
