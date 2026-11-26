"""Metrics shared by the experiments (all torch, no model calls)."""

import torch


def next_token_kl(ref_log_probs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """KL(P_ref || P_x) in nats for reference log-probs [R, V] and logits [B, V] -> [R, B] (float64)."""
    log_q = torch.log_softmax(logits.double(), dim=-1)
    p = ref_log_probs.double().exp()
    return (p * ref_log_probs.double()).sum(-1, keepdim=True) - p @ log_q.T


class NormRatio:
    """
    Contraction / expansion of displacements between a source layer and an output layer, around the unsteered
    activations x (source) and f(x) (output). Fitted on one plane's N grid points h_i:

        s_src = sqrt(mean_i ||h_i - x||^2),   s_out = sqrt(mean_i ||f(h_i) - f(x)||^2),
        log10 R(h) = log10( (||f(h) - f(x)|| / s_out) / (||h - x|| / s_src) ).

    Dividing by the spreads removes the overall gain of the map: log10 R < 0 means the point's displacement shrinks
    more than the plane's average (contraction), > 0 that it grows more (expansion). NaN at h = x.
    """

    def __init__(self, x_src: torch.Tensor, x_out: torch.Tensor, h_src: torch.Tensor, h_out: torch.Tensor):
        """x_src [d], x_out [J, d]; h_src [N, d], h_out [N, J, d] (J output positions)."""
        self.x_src, self.x_out = x_src.double(), x_out.double()
        self.s_src = (h_src.double() - self.x_src).norm(dim=-1).pow(2).mean().sqrt()
        self.s_out = (h_out.double() - self.x_out).norm(dim=-1).pow(2).mean(0).sqrt()  # [J]

    def __call__(self, h_src: torch.Tensor, h_out: torch.Tensor) -> torch.Tensor:
        """h_src [N, d], h_out [N, J, d] -> log10 R [N, J] float32."""
        r_src = (h_src.double() - self.x_src).norm(dim=-1) / self.s_src
        r_out = (h_out.double() - self.x_out).norm(dim=-1) / self.s_out
        lr = torch.log10(r_out) - torch.log10(r_src)[:, None]
        return lr.masked_fill((r_src == 0)[:, None], float('nan')).float()


def project_onto(vectors: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
    """Coordinates [..., k] of vectors [..., d] along axes [k, d], in units of each axis (s.a / ||a||^2)."""
    return vectors.double() @ (axes.double() / axes.double().pow(2).sum(-1, keepdim=True)).T
