import jax.numpy as jnp
import flax.nnx as nnx

from lifting import ads_distance_xz


def segment_mean(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
):
    """
    Compute the mean of data for each segment.

    :param data: The data to be aggregated.
    :param segment_ids: An integer array of segment IDs.
    :param num_segments: The number of segments.
    :param indices_are_sorted: Whether the segment IDs are already sorted.
    :param unique_indices: Whether the segment IDs are unique.

    :return: The mean of data for each segment.
    """
    nominator = jax.ops.segment_sum(
        data,
        segment_ids,
        num_segments,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
    )
    denominator = jax.ops.segment_sum(
        jnp.ones_like(data),
        segment_ids,
        num_segments,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
    )
    return nominator / jnp.maximum(
        denominator, jnp.ones(shape=[], dtype=denominator.dtype)
    )


class AdS_GCL(nnx.Module):
    """
    CG Equivariant Convolutional Layer
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edge_attr_nf=0,
        act_fn=nnx.silu,
        residual=True,
        coords_agg="mean",
        *,
        rngs
    ):
        super().__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.coords_agg = coords_agg
        self.epsilon = 1e-8
        self.edge_attr_nf = 1 + edge_attr_nf

        self.edge_mlp = nnx.Sequential(
            nnx.Linear(
                input_edge + self.edge_attr_nf,
                hidden_nf,
                rngs=rngs,
                bias_init=nnx.initializers.constant(0.1),
            ),
            act_fn,
            nnx.Linear(
                hidden_nf,
                hidden_nf,
                rngs=rngs,
                bias_init=nnx.initializers.constant(0.1),
            ),
            act_fn,
        )

        self.node_mlp = nnx.Sequential(
            nnx.Linear(
                hidden_nf + input_nf,
                hidden_nf,
                rngs=rngs,
                bias_init=nnx.initializers.constant(0.1),
            ),
            act_fn,
            nnx.Linear(
                hidden_nf,
                output_nf,
                rngs=rngs,
                bias_init=nnx.initializers.constant(0.1),
            ),
        )

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is not None:
            out = jnp.concatenate([source, target, radial, edge_attr], axis=1)
        else:
            out = jnp.concatenate([source, target, radial], axis=1)
        out = self.edge_mlp(out)
        return out

    def node_model(self, h, edge_index, edge_attr):
        row, _ = edge_index
        agg = segment_mean(edge_attr, row, num_segments=h.shape[0])
        agg = jnp.concatenate([h, agg], axis=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = h + out
        return out

    def coord2dist(self, edge_index, xz):
        row, col = edge_index
        dist = ads_distance_xz(xz[row], xz[col])[:, None]
        return dist

    def __call__(self, xz, h, edge_index, edge_attr=None):
        row, col = edge_index
        dist = self.coord2dist(edge_index, xz)
        edge_feat = self.edge_model(h[row], h[col], dist, edge_attr)
        h = self.node_model(h, edge_index, edge_feat)

        return h


class AdSGNN(nnx.Module):
    def __init__(
        self,
        in_node_nf,
        hidden_nf,
        out_node_nf,
        edge_attr_nf=0,
        act_fn=nnx.silu,
        n_layers=4,
        residual=True,
        *,
        rngs
    ):
        """
        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param edge_attr_nf: Number of features for the edge attributes (default=0, no edge attributes)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the cggnn
        :param residual: Use residual connections, we recommend not changing this one
        """
        super().__init__()
        self.act_fn = act_fn
        self.hidden_nf = hidden_nf
        self.edge_attr_nf = edge_attr_nf
        self.n_layers = n_layers
        out_node_nf = out_node_nf if out_node_nf is not None else hidden_nf
        self.embedding_in = nnx.Linear(in_node_nf, self.hidden_nf, rngs=rngs)
        self.embedding_out = nnx.Linear(self.hidden_nf, out_node_nf, rngs=rngs)
        self.layers = []
        for i in range(0, n_layers):
            self.layers.append(
                AdS_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    self.edge_attr_nf,
                    act_fn=act_fn,
                    residual=residual,
                    rngs=rngs,
                )
            )

    def __call__(self, pos, x, edge_index, edge_attr):
        h = self.embedding_in(x)
        for layer in self.layers:
            h = layer(pos, h, edge_index, edge_attr)
        h = self.embedding_out(h)
        return h


if __name__ == "__main__":
    import jax 

    rngs = nnx.Rngs(0)
    model = AdSGNN(64, 64, 64, n_layers=4, rngs=rngs)

    pos = jax.random.uniform(rngs.keys(), (10, 3))
    x = jax.random.uniform(rngs.keys(), (10, 64))
    edge_index = jax.random.randint(rngs.keys(), (2, 100), 0, 10)

    h = model(pos, x, edge_index, None)
    print(h.shape)