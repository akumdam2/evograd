"""Level-4 workloads: whole models, executed the way training executes them.

Levels 1-3 are declared operators -- a forward reference, a pair contract, a
shape suite -- because that is what an evolved kernel replaces. Level 4 is the
other end of the telescope: one real model, one real training step, run through
the framework a user would actually run. Nothing here is declared through
``OpDecl``; a model is not an operator and forcing it into that shape would
distort both.

What Level 4 exists to provide is a *reference execution* that later stages can
observe. The operator suites answer "is this kernel faster than that kernel";
only a real training step can answer "does this kernel appear in the model at
all, at which shapes, and how often".
"""
