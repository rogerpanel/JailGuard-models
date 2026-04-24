"""Type-specific ODE dynamics for the heterogeneous interaction graph.

Implements Eq. (4) of the manuscript:

    dh_v(t)/dt = f_{τ(v)}(h_v(t), Σ_{(u,r) ∈ N(v,t)} α_{v,u,r}(t) · g_r(h_u(t)))

Each f_τ is a two-layer MLP with spectral normalization, one per node
type τ ∈ {U, S, Q, M}. The integration is performed by torchdiffeq's
dopri5 solver with adjoint sensitivity for O(1) memory training.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

try:  # torchdiffeq is an optional heavy dependency.
    from torchdiffeq import odeint, odeint_adjoint
except ImportError:  # pragma: no cover — CI will install it from requirements.
    odeint = odeint_adjoint = None

from ct_dgnn.models.spectral_norm import SpectralMLP


class TypeSpecificMLP(nn.Module):
    """Two-layer MLP f_τ : R^d × R^d → R^d used as the ODE RHS.

    The spectral normalization yields per-matrix 1-Lipschitz linear
    mappings; the composed MLP is bounded by the product of those
    (Theorem 1 in the manuscript).
    """

    def __init__(self, embed_dim: int, hidden_dim: int | None = None,
                 dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        # Input: [h_v ‖ aggregated_msg] of length 2d
        self.mlp = SpectralMLP(
            dims=[2 * embed_dim, hidden_dim, embed_dim],
            activation=nn.SiLU(),
            dropout=dropout,
        )

    def forward(self, h_v: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([h_v, message], dim=-1))

    @torch.no_grad()
    def lipschitz_constant(self) -> torch.Tensor:
        return self.mlp.lipschitz_constant()


class HeterogeneousODEDynamics(nn.Module):
    """Bundle of type-specific f_τ modules, one per node type."""

    def __init__(self, node_types: list[str], embed_dim: int, hidden_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.node_types = node_types
        self.f_tau: Dict[str, TypeSpecificMLP] = nn.ModuleDict(
            {t: TypeSpecificMLP(embed_dim, hidden_dim, dropout) for t in node_types}
        )

    def forward(self, h: torch.Tensor, message: torch.Tensor,
                node_type_ids: torch.Tensor) -> torch.Tensor:
        """Compute dh/dt for a batch of nodes.

        Parameters
        ----------
        h : (N, d)
        message : (N, d)
        node_type_ids : (N,) integer tensor indexing into `self.node_types`.
        """
        dh = torch.zeros_like(h)
        for i, t in enumerate(self.node_types):
            mask = node_type_ids == i
            if mask.any():
                dh[mask] = self.f_tau[t](h[mask], message[mask])
        return dh

    @torch.no_grad()
    def lipschitz_constant(self) -> torch.Tensor:
        Ls = [m.lipschitz_constant() for m in self.f_tau.values()]
        return torch.stack(Ls).max()


class NeuralODEBlock(nn.Module):
    """Wrap dynamics + attention + message passing into a closure solvable
    by torchdiffeq. Between two event times (t_i, t_{i+1}) the topology and
    attention coefficients are frozen; only h evolves continuously.
    """

    def __init__(
        self,
        dynamics: HeterogeneousODEDynamics,
        solver: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4,
        use_adjoint: bool = True,
    ) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = use_adjoint
        self._message: torch.Tensor | None = None
        self._types: torch.Tensor | None = None

    def set_step_inputs(self, message: torch.Tensor,
                         node_type_ids: torch.Tensor) -> None:
        self._message = message
        self._types = node_type_ids

    # torchdiffeq requires a callable (t, y) -> dy/dt
    def _rhs(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.dynamics(h, self._message, self._types)

    def integrate(self, h0: torch.Tensor,
                  t_span: torch.Tensor) -> torch.Tensor:
        if odeint is None:
            raise RuntimeError("torchdiffeq not installed")
        solver = odeint_adjoint if self.use_adjoint else odeint
        traj = solver(
            self._rhs, h0, t_span,
            method=self.solver, rtol=self.rtol, atol=self.atol,
            adjoint_params=tuple(self.dynamics.parameters()) if self.use_adjoint else None,
        )
        return traj[-1]
