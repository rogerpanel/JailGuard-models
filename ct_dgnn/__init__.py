"""CT-DGNN-JailGuard: continuous-time heterogeneous dynamic graph
neural network with certified robustness for LLM-jailbreak campaign
detection on API interaction graphs."""

__version__ = "1.0.0"

# Lazy attribute access so pure-Python helpers (metrics, certificate
# math, config loading) remain importable without PyTorch installed.
_LAZY = {
    "CTDGNNJailGuard":           ("ct_dgnn.models.ct_dgnn",        "CTDGNNJailGuard"),
    "APIInteractionGraphBuilder": ("ct_dgnn.data.graph_builder",   "APIInteractionGraphBuilder"),
    "certified_radius":          ("ct_dgnn.robustness.certificate","certified_radius"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(name)


__all__ = ["CTDGNNJailGuard", "APIInteractionGraphBuilder",
           "certified_radius", "__version__"]
