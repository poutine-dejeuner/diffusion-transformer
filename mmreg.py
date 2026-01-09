import enum
import numpy as np
import torch

def mmreg(x_nn_embed, x_topo_embed_dist, alpha=1.0):
    eps = 1e-8
    N = x_nn_embed.shape[0]
    x_flat = x_nn_embed.reshape(N, -1)
    xdist = torch.cdist(x_flat, x_flat)
    xdistnorm = torch.norm(xdist, p="fro") 
    xdist = xdist / (xdistnorm + eps)
    x_embeddist = x_topo_embed_dist
    x_embeddistnorm = torch.norm(x_embeddist, p="fro")
    x_embeddist = x_embeddist / (x_embeddistnorm + eps)

    reg_term = alpha * (1 / N ** 2) * torch.norm(xdist - x_embeddist, p="fro")

    return reg_term


def manifold_matching_reg(x, x_embed, alpha=1.0):
    """
    Compute a manifold matching regularization term between original data and embeddings.
    This encourages the embedding to preserve local neighborhood structure using Euclidean distances.

    Args:
        x: Original data tensor of shape (N, D1)
        x_embed: Embedded data tensor of shape (N, D2)
        alpha: Regularization weight

    Returns:
        Scalar tensor representing the regularization loss
    """
    eps = 1e-8
    N = x.shape[0]
    x_flat = x.reshape(N, -1)
    xdist = torch.cdist(x_flat, x_flat)
    xdistnorm = torch.norm(xdist, p="fro") 
    xdist = xdist / (xdistnorm + eps)
    x_embeddist = torch.cdist(x_embed, x_embed)
    x_embeddistnorm = torch.norm(x_embeddist, p="fro")
    x_embeddist = x_embeddist / (x_embeddistnorm + eps)
    if not torch.is_nonzero(xdistnorm): 
        np.save("buggy_x.npy", x.detach().cpu().numpy())
        breakpoint()
    if not torch.is_nonzero(x_embeddistnorm):
        np.save("buggy_xemb.npy", x_embed.detach().cpu().numpy())
        breakpoint()
    reg_term = alpha * (1 / N ** 2) * torch.norm(xdist - x_embeddist, p="fro")

    return reg_term


class TopoAlgoType(enum.Enum):
    """
    Which type of topological data analysis algorithm to use.
    """
    PCA = enum.auto()
    UMAP = enum.auto()
    TSNE = enum.auto()
