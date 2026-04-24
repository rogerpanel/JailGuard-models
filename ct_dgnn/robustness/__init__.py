"""Robustness utilities. Only `certificate` is pure-Python; the other
modules import torch lazily so the certificate math stays usable in
light-weight environments."""

from ct_dgnn.robustness.certificate import (
    certified_radius,
    gronwall_bound,
)

_LAZY = {
    "jacobian_frobenius":         ("ct_dgnn.robustness.jacobian_reg", "jacobian_frobenius"),
    "model_lipschitz_constants":  ("ct_dgnn.robustness.lipschitz",    "model_lipschitz_constants"),
    "pgd_evaluate":               ("ct_dgnn.robustness.pgd_attack",   "pgd_evaluate"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(name)


__all__ = [
    "certified_radius",
    "gronwall_bound",
    "jacobian_frobenius",
    "model_lipschitz_constants",
    "pgd_evaluate",
]
