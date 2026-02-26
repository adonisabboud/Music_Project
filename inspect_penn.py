import penn
import inspect

print("penn.infer signature:", inspect.signature(penn.infer))
print("penn.from_audio signature:", inspect.signature(penn.from_audio))
try:
    from penn import core
    print("penn.core.preprocess signature:", inspect.signature(core.preprocess))
except ImportError:
    print("Could not import penn.core")
