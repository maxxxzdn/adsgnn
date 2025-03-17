import jax
import jax.numpy as jnp
from functools import partial

from com import compute_ads_com


def ads_distance_xz(xz: jnp.ndarray, xzp: jnp.ndarray) -> jnp.ndarray:
    """
    xz: jnp.ndarray of shape (N, d+1)
    xzp: jnp.ndarray of shape (N, d+1)
    """
    r = jnp.linalg.norm(xz[..., :-1] - xzp[..., :-1], axis=-1)
    z, zp = xz[..., -1], xzp[..., -1]
    return (z**2 + zp**2 + r**2) / (2 * z * zp)


@partial(jax.jit, static_argnums=(1, 2))
def nearest_neighbors(
    x: jnp.array,
    k: int,
    include_self: bool = True,
):
    """Returns k nearest neighbors of each node in x. Nearest in terms of Euclidean distance."""
    n_nodes = x.shape[0]
    dr = x[:, None, :] - x[None, :, :]
    distance_matrix = jnp.sum(dr**2, axis=-1)

    if not include_self:
        distance_matrix = (
            distance_matrix + jnp.eye(x.shape[0]) * 1e10
        )  # Set diagonal to inf to mask out self

    sources = jnp.repeat(jnp.arange(n_nodes), k)
    targets = jnp.argsort(distance_matrix, axis=-1)[:, :k].ravel()
    return sources, targets


@partial(jax.jit, static_argnums=(1, 2))
def ads_nearest_neighbors(
    xz: jnp.array,
    k: int,
    include_self: bool = True,
):
    """Returns k nearest neighbors of each node in x. Nearest in terms of AdS distance."""
    n_nodes = xz.shape[0]
    distance_matrix = ads_distance_xz(xz[:, None, :], xz[None, :, :])

    if not include_self:
        distance_matrix = (
            distance_matrix + jnp.eye(n_nodes) * 1e10
        )  # Set diagonal to inf to mask out self

    sources = jnp.repeat(jnp.arange(n_nodes), k)
    targets = jnp.argsort(distance_matrix, axis=-1)[:, :k].ravel()
    return sources, targets


def lift_to_ads_A(x, h, k, z0=1e-6, **kwargs):
    """
    Lifting A: find NNs (Euclidean) at the boundary, compute COM of each cluster, assign original features to the COMs.
    Output: x of COMs, z of COMs, h original.
    """
    # to each point assign z-coordinate (boundary)
    z = jnp.array([z0] * x.shape[0])[:, None]

    # find nearest neighbors at the boundary
    _, clusters = nearest_neighbors(x, k, False)
    clusters_x = x[clusters.reshape(-1, k)]
    clusters_z = z[clusters.reshape(-1, k)]

    # compute ads com for each cluster
    x_emb, z_emb = jax.vmap(compute_ads_com)(clusters_x, clusters_z)
    return x_emb.squeeze(1), z_emb.squeeze(1), h


def lift_to_ads_B(x, h, k, z0=1e-6, **kwargs):
    """
    Lifting B: find NNs (Euclidean) at the boundary, compute COM of each cluster, interpolate features to the COMs.
    Output: x original, z of COMs, h of COMs.
    """
    # to each point assign z-coordinate (boundary)
    z = jnp.array([z0] * x.shape[0])[:, None]

    # find nearest neighbors at the boundary
    _, clusters = nearest_neighbors(x, k, False)
    clusters_x = x[clusters.reshape(-1, k)]
    clusters_z = z[clusters.reshape(-1, k)]

    # compute ads com for each cluster
    _, z_emb = jax.vmap(compute_ads_com)(clusters_x, clusters_z)
    return x, z_emb.squeeze(1), h


def lift_to_ads_C(x, h, k, z0=1e-6, **kwargs):
    """
    Lifting C: find NNs (Euclidean) at the boundary, compute COM of each cluster, interpolate features to the COMs.
    Output: x of COMs, z of COMs, h of COMs.
    """
    # to each point assign z-coordinate (boundary)
    z = jnp.array([z0] * x.shape[0])[:, None]

    # find nearest neighbors at the boundary
    centers, clusters = nearest_neighbors(x, k, False)

    clusters_x = x[clusters.reshape(-1, k)]
    clusters_z = z[clusters.reshape(-1, k)]

    # compute ads com for each cluster
    x_emb, z_emb = jax.vmap(compute_ads_com)(clusters_x, clusters_z)
    x_emb, z_emb = x_emb.squeeze(1), z_emb.squeeze(1)

    # interpolate feautures from original x, z to x_emb, z_emb
    xz_emb_centers = jnp.concatenate([x_emb, z_emb], axis=1)[centers]
    xz_emb_clusters = jnp.concatenate([x_emb, z_emb], axis=1)[clusters]

    distances = ads_distance_xz(xz_emb_centers, xz_emb_clusters).reshape(-1, k)

    # interpolation
    weights = 1 / distances
    denom = jnp.sum(weights, axis=1).reshape(-1, 1)
    weights = weights / denom
    h_emb = jnp.sum(weights[..., None] * h[clusters].reshape(-1, k, h.shape[1]), axis=1)

    return x_emb, z_emb, h_emb


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def lift_point_cloud(
    x: jnp.ndarray,
    h: jnp.ndarray,
    k_lifting: int,
    k_edge_index: int,
    lift_x: bool,
    lift_h: bool,
) -> jnp.ndarray:
    """
    Lifts a point cloud from Euclidean space to the AdS space.

    Args:
        x: positions of nodes in the point cloud.
        h: features of nodes in the point cloud.
        k_lifting: number of neighbors to consider for lifting.
        k_edge_index: number of neighbors to consider for connectivity.
        lift_x: whether to lift positions.
        lift_h: whether to lift features.

    Returns:
        x: positions of nodes in the lifted point cloud.
        z: z-coordinates of the lifted point cloud.
        h: features of nodes in the lifted point cloud.
        edge_index: connectivity in and within the lifted point cloud.
    """
    if lift_x and lift_h:
        x, z, h = lift_to_ads_C(x, h, k_lifting)  # x, z, h of COMs
    elif lift_x:
        x, z, h = lift_to_ads_B(x, h, k_lifting)  # x, z of COMs, h original
    else:
        x, z, h = lift_to_ads_A(x, h, k_lifting)  # x original, z of COMs, h original

    if k_edge_index > 0:
        edge_index = ads_nearest_neighbors(
            jnp.concatenate([x, z], axis=1), k_edge_index, include_self=True
        )
    else:
        edge_index = None

    return x, z, h, edge_index
