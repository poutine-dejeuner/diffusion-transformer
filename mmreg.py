from pathlib import Path
import enum
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import TensorDataset, Dataset, DataLoader
from sklearn.metrics import pairwise_distances


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

    # compute x pairwise distance matrix
    xdist = torch.cdist(x_flat, x_flat)
    xdistnorm = torch.norm(xdist, p="fro")
    xdist = xdist / (xdistnorm + eps)

    # compute x_embed pairwise data matrix
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


def topo_representation(data: np.ndarray, algo: TopoAlgoType, n_dimensions:
                        int = 2) -> np.ndarray:
    """
    Generate topological data augmentation using the specified algorithm.

    Parameters:
    data (np.ndarray): The input data to augment.
    algo (TopoAlgoType): The topological data analysis algorithm to use.
    n_dimensions (int): Number of dimensions for the output data.
    """

    N = data.shape[0]
    data = np.reshape(data, (N, -1))

    if algo == TopoAlgoType.PCA:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_dimensions)
        transformed_data = pca.fit_transform(data)
    elif algo == TopoAlgoType.UMAP:
        import umap
        reducer = umap.UMAP(n_components=n_dimensions)
        transformed_data = reducer.fit_transform(data)
    elif algo == TopoAlgoType.TSNE:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=n_dimensions)
        transformed_data = tsne.fit_transform(data)

    # compute the pairwise distances in the transformed space
    distances = pairwise_distances(transformed_data)

    return distances


class PadToDivisible(nn.Module):
    """
    Pad image to make dimensions divisible by a given divisor.
    Useful for models that require specific dimension constraints (e.g., VAE with divisor=8).

    Args:
        divisor: The number by which dimensions should be divisible (default: 8)
        fill: Pixel fill value for padding (default: 0)
        padding_mode: Padding mode - 'constant', 'edge', 'reflect', 'symmetric' (default: 'constant')
    """

    def __init__(self, divisor=8, fill=0, padding_mode='constant'):
        super().__init__()
        self.divisor = divisor
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        """
        Args:
            img: PIL Image or Tensor

        Returns:
            Padded PIL Image or Tensor
        """
        if isinstance(img, torch.Tensor):
            h, w = img.shape[-2:]
            is_tensor = True
        else:
            w, h = img.size
            is_tensor = False

        # Calculate padding needed
        pad_h = (self.divisor - h % self.divisor) % self.divisor
        pad_w = (self.divisor - w % self.divisor) % self.divisor

        # Distribute padding evenly on both sides
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        if is_tensor:
            # Use torch padding for tensors
            padding = (pad_left, pad_right, pad_top, pad_bottom)
            return torch.nn.functional.pad(img, padding, mode=self.padding_mode, value=self.fill)
        else:
            # Use PIL padding for images
            padding = (pad_left, pad_top, pad_right, pad_bottom)
            return transforms.functional.pad(img, padding, fill=self.fill, padding_mode=self.padding_mode)

    def __repr__(self):
        return f"{self.__class__.__name__}(divisor={self.divisor}, fill={self.fill}, padding_mode='{self.padding_mode}')"


def dataset_to_array(dataset, max_samples=None):
    """
    Convertit un dataset torchvision en numpy array.

    Args:
        dataset: Dataset torchvision
        max_samples: Nombre maximum d'échantillons (None = tous)

    Returns:
        data: Numpy array contenant les données du dataset
    """
    n_samples = len(dataset) if max_samples is None else min(
        max_samples, len(dataset))

    data_list = []
    for i in range(n_samples):
        item = dataset[i]
        if isinstance(item, tuple):
            data_list.append(item[0])
        else:
            data_list.append(item)

    data = torch.stack(data_list)
    return data.cpu().numpy()


def get_dataset_and_topo_repr(dataset, dtype, topo_algo: TopoAlgoType,
                              n_components: int):
    data = dataset_to_array(dataset)
    topo_repr = topo_representation(data, topo_algo, n_components)
    topo_repr = torch.from_numpy(topo_repr).to(dtype=dtype)
    data = torch.from_numpy(data).to(dtype=dtype)
    topo_repr = torch.cdist(topo_repr, topo_repr)

    # pad
    pad_fn = PadToDivisible(divisor=8)
    data = pad_fn(data)

    dataset = TensorDataset(data)
    return dataset, topo_repr
