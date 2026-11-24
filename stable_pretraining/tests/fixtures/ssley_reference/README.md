# Pinned SSL-EY reference fixtures

Unmodified loss, LARS and scheduler from jameschapman19/solo-learn commit
`11eac929a3cfb8fe28cefa573ef87e08fcf1eb27` (branch `ssley`). Source paths:
`solo/losses/ssley.py`, `solo/utils/lars.py`, `solo/utils/lr_scheduler.py`.
Loss fixture's solo-learn gather import is removed by AST in the test harness;
identity gather matches the released single-GPU CIFAR configuration.
The model harness independently transcribes the constructor/forward from
`solo/methods/base.py` and `solo/methods/ssley.py`.
Original MIT notices are retained; the scheduler is copied by solo-learn from
Lightning Bolts (Apache 2.0). See the sibling LICENSE files.
These fixtures are test oracles, not a dependency on the solo-learn package.
