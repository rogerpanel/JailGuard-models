from ct_dgnn.utils.config import load_config

_LAZY = {"set_seed": ("ct_dgnn.utils.seed", "set_seed")}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(name)


__all__ = ["load_config", "set_seed"]
