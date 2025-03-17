import jax
import jax.numpy as jnp


def get_NY0(x: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """
    x: jnp.ndarray of shape (N, d)
    z: jnp.ndarray of shape (N, 1)
    """
    num_nodes = x.shape[0]
    sum_term = lambda xi, zi: zi / 2 * (1 + 1 / (zi**2) * (1 + jnp.dot(xi, xi)))
    return 1 / num_nodes * jax.vmap(sum_term)(x, z).sum(keepdims=True)


def get_NYa(x: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """
    x: jnp.ndarray of shape (N, d)
    z: jnp.ndarray of shape (N, 1)
    """
    num_nodes = x.shape[0]
    sum_term = lambda xi, zi: xi / zi
    return 1 / num_nodes * jax.vmap(sum_term)(x, z).sum(0, keepdims=True)


def get_NYdp1(x: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """
    x: jnp.ndarray of shape (N, d)
    z: jnp.ndarray of shape (N, 1)
    """
    num_nodes = x.shape[0]
    sum_term = lambda xi, zi: zi / 2 * (1 - 1 / zi**2 * (1 - jnp.dot(xi, xi)))
    return 1 / num_nodes * jax.vmap(sum_term)(x, z).sum(keepdims=True)


def get_N_from_Y(NY0: jnp.ndarray, NYa: jnp.ndarray, NYdp1: jnp.ndarray) -> jnp.ndarray:
    """
    NY0: jnp.ndarray of shape (1,1)
    NYa: jnp.ndarray of shape (1,d)
    NYdp1: jnp.ndarray of shape (1,1)
    """
    return jnp.sqrt(NY0**2 - (NYa**2).sum() - NYdp1**2)


def rescale_NY(NY0: jnp.ndarray, NYa: jnp.ndarray, NYdp1: jnp.ndarray) -> jnp.ndarray:
    """
    NY0: jnp.ndarray of shape (1,1)
    NYa: jnp.ndarray of shape (1,d)
    NYdp1: jnp.ndarray of shape (1,1)
    """
    N = get_N_from_Y(NY0, NYa, NYdp1)
    return NY0 / N, NYa / N, NYdp1 / N


def get_z_bar(Ya: jnp.ndarray, Ydp1: jnp.ndarray) -> jnp.ndarray:
    """
    Ya: jnp.ndarray of shape (1,d)
    Ydp1: jnp.ndarray of shape (1,1)
    """
    return (Ydp1 + jnp.sqrt(1 + Ydp1**2 + jnp.dot(Ya, Ya.T))) / (
        1 + jnp.dot(Ya, Ya.T)
    )


def get_x_bar(z_bar: jnp.ndarray, Ya: jnp.ndarray) -> jnp.ndarray:
    """
    z_bar: jnp.ndarray of shape (1,)
    Ya: jnp.ndarray of shape (d,)
    """
    return z_bar * Ya


def compute_ads_com(x: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """
    x: jnp.ndarray of shape (N, d)
    z: jnp.ndarray of shape (N, 1)
    """
    NY0, NYa, NYdp1 = get_NY0(x, z), get_NYa(x, z), get_NYdp1(x, z)
    NY0, NYa, NYdp1 = rescale_NY(NY0, NYa, NYdp1)
    z_bar = get_z_bar(NYa, NYdp1)
    x_bar = get_x_bar(z_bar, NYa)
    return x_bar, z_bar
