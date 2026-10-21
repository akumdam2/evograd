"""Model integrations of the verification gates.

Each module here connects one benchmark model to the generic gates: it reads
the model's task declarations and harvest snapshot, never edits them, and owns
what only that model knows (how to record its real training calls).
"""
