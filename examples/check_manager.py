import stable_pretraining as spt
import inspect

print("Package location:", spt.__file__)
print("Manager:", spt.Manager)
print("Manager type:", type(spt.Manager))
print("Is Manager callable:", callable(spt.Manager))

if callable(spt.Manager):
    print("Signature:", inspect.signature(spt.Manager))
