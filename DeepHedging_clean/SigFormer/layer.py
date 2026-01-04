from functools import partial
from typing import List, Optional

import einops
import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.random import PRNGKey
from jaxtyping import Array, Float
from .utils import split_key
from signax import signature
from signax.tensor_ops import mult_fused_restricted_exp, restricted_exp


class TensorLinear(eqx.Module):

    lin: nn.Linear
    n_heads: int = eqx.field(static=True)

    def __init__(
        self,
        in_features,
        out_features,
        order,
        n_heads=1,
        use_bias=True,
        *,
        key: PRNGKey,
    ):
        assert order >= 1
        self.n_heads = n_heads
        self.lin = nn.Linear(
            in_features**order,
            n_heads * out_features**order,
            use_bias=use_bias,
            key=key,
        )

    def __call__(
        self,
        x: Float[Array, " dim"],
        *,
        key: Optional[PRNGKey] = None,
    ) -> Float[Array, " out_dim"]:
        """Project tensorized inputs per head.

        Args:
            x: Flattened signature slice shaped (dim,).
            key: Optional PRNG key for linear layer.

        Returns:
            Array: Head-stacked projections shaped (n_heads, out_dim).
        """

        x = self.lin(x, key=key)

        x = einops.rearrange(
            x, "(n_heads out_dim) -> n_heads out_dim", n_heads=self.n_heads
        )
        return x


class TensorLinearOutput(eqx.Module):

    lin: nn.Linear

    def __init__(
        self,
        in_features,
        out_features,
        order,
        n_heads=1,
        use_bias=True,
        *,
        key: PRNGKey,
    ):

        self.lin = nn.Linear(
            n_heads * in_features**order,
            out_features**order,
            use_bias=use_bias,
            key=key,
        )

    def __call__(
        self,
        x: Float[Array, "n_heads dim"],
        *,
        key: Optional[PRNGKey] = None,
    ) -> Float[Array, " out_dim"]:
        """Merge head outputs and map back to output order.

        Args:
            x: Head-stacked features shaped (n_heads, dim).
            key: Optional PRNG key for linear layer.

        Returns:
            Array: Output projection shaped (out_dim,).
        """

        x = einops.rearrange(x, "... -> (...)")
        x = self.lin(x)

        return x


class SelfAttentionAtDepth(eqx.Module):

    query_proj: TensorLinear
    key_proj: TensorLinear
    value_proj: TensorLinear
    output_proj: TensorLinear

    dropout: nn.Dropout

    n_heads: int = eqx.field(static = True)

    def __init__(
        self,
        order: int,
        dim: int,
        n_heads: int,
        dropout: float = 0.0,
        *,
        key: jrandom.PRNGKey,
    ):
        qkey, kkey, vkey, okey = jrandom.split(key, 4)

        query_size = key_size = value_size = output_size = dim
        self.n_heads = n_heads
        self.query_proj = TensorLinear(
            in_features=query_size,
            out_features=query_size,
            order=order,
            n_heads=n_heads,
            use_bias=False,
            key=qkey,
        )

        self.key_proj = TensorLinear(
            in_features=key_size,
            out_features=key_size,
            order=order,
            n_heads=n_heads,
            use_bias=False,
            key=kkey,
        )

        self.value_proj = TensorLinear(
            in_features=value_size,
            out_features=value_size,
            order=order,
            n_heads=n_heads,
            key=vkey,
        )

        self.output_proj = TensorLinearOutput(
            in_features=output_size,
            out_features=output_size,
            order=order,
            n_heads=n_heads,
            key=okey,
        )

        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        x: Float[Array, "seq_len dim"],
        *,
        key: PRNGKey = None,
    ) -> Float[Array, "seq_len dim"]:
        """Apply depth-specific self-attention over a sequence.

        Args:
            x: Input sequence of embeddings (seq_len, dim).
            key: Optional PRNG key for dropout/attention.

        Returns:
            Array: Attended sequence with same shape.
        """

        shape = x.shape
        seq_len = shape[0]
        x = einops.rearrange(x, "seq_len ... -> seq_len (...)")

        q = jax.vmap(self.query_proj)(x)
        k = jax.vmap(self.key_proj)(x)
        v = jax.vmap(self.value_proj)(x)

        mask = jnp.tril(jnp.ones((seq_len, seq_len)))

        attn_fn = partial(
            eqx.nn._attention.dot_product_attention,
            dropout=self.dropout,
            mask=mask,
            inference=None,
        )

        keys = None if key is None else jrandom.split(key, self.n_heads)

        # q = einops.rearrange(q, "seq_len n_heads dim -> dim n_heads seq_len")
        # k = einops.rearrange(k, "seq_len n_heads dim -> dim n_heads seq_len")
        # v = einops.rearrange(v, "seq_len n_heads dim -> dim n_heads seq_len")

        x = jax.vmap(attn_fn, in_axes=1, out_axes=1)(q, k, v, key=keys)

        # x = einops.rearrange(x, "dim num_heads seq_len -> seq_len num_heads dim")
        x = jax.vmap(self.output_proj)(x)

        x = jnp.reshape(x, shape)

        return x


class TensorSelfAttention(eqx.Module):

    all_attn: List[SelfAttentionAtDepth]

    def __init__(
        self,
        order: int,
        dim: int,
        n_heads: int,
        dropout: float = 0.0,
        *,
        key: jrandom.PRNGKey,
    ) -> None:

        all_attn = []
        for i in range(order):
            attn = SelfAttentionAtDepth(
                order=i + 1,
                dim=dim,
                n_heads=n_heads,
                dropout=dropout,
                key=jrandom.fold_in(key, i + 1),
            )
            all_attn.append(attn)

        self.all_attn = all_attn

    def __call__(
        self,
        x: List[Array],
        *,
        key: jrandom.PRNGKey = None,
    ) -> List[Array]:
        """Run stacked attention blocks across signature levels.

        Args:
            x: List of tensors for each signature level.
            key: Optional PRNG key to split per block.

        Returns:
            list[Array]: Attended tensors per level.
        """

        result = []
        for i, (attn, xx) in enumerate(zip(self.all_attn, x)):
            key = split_key(key)
            attn_x = attn(x=xx, key=key)
            result.append(attn_x)

        return result


class TensorLayerNorm(eqx.Module):

    norms: List[nn.LayerNorm]

    def __init__(self, dim: int, order: int):
        self.norms = [nn.LayerNorm((dim,) * i) for i in range(1, order + 1)]

    def __call__(self, x: List[Array]) -> List[Array]:
        """Apply LayerNorm to each level independently.

        Args:
            x: List of tensors across signature depths.

        Returns:
            list[Array]: Normalized tensors per level.
        """
        out = []
        for norm, xx in zip(self.norms, x):
            # xx has shape (seq_len, dim)
            # Apply LayerNorm row-wise
            xx = jax.vmap(norm)(xx)
            out.append(xx)
        return out


class TensorDropout(eqx.Module):

    dropout: nn.Dropout

    def __init__(
        self,
        dropout_p=0.0,
    ) -> None:
        self.dropout = nn.Dropout(dropout_p)

    def __call__(self, x: List[Array], *, key=None) -> List[Array]:
        """Apply dropout to each tensor in the list."""
        key = [None] * len(x) if key is None else jrandom.split(key, len(x))
        return [self.dropout(xx, key=kk) for xx, kk in zip(x, key)]


class TensorMLP(eqx.Module):

    ff: List[nn.Sequential]

    def __init__(self, dim: int, order: int, d_ff: int, dropout=0.0, *, key: PRNGKey):
        self.ff = [
            nn.Sequential(
                layers=[
                    nn.Linear(dim**i, d_ff, key=jrandom.fold_in(key, i * 2)),
                    nn.Lambda(jax.nn.gelu),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, dim**i, key=jrandom.fold_in(key, i * 2 + 1)),
                ]
            )
            for i in range(1, order + 1)
        ]

    def __call__(
        self,
        x: List[Array],
        *,
        key: PRNGKey = None,
    ) -> List[Array]:
        """Feed-forward network per signature level with reshape safety.

        Args:
            x: List of tensors per level.
            key: Optional PRNG key split per level.

        Returns:
            list[Array]: Transformed tensors matching input shapes.
        """
        shapes = [xx.shape for xx in x]
        x = [einops.rearrange(xx, "... -> (...)") for xx in x]
        key = [None] * len(x) if key is None else jrandom.split(key, len(x))
        x = [ff(xx, key=kk) for ff, xx, kk in zip(self.ff, x, key)]
        x = [jnp.reshape(xx, shape) for xx, shape in zip(x, shapes)]
        return x


class TensorAdd(eqx.Module):
    def __call__(self, x: List[Array], y: List[Array]) -> List[Array]:
        """Elementwise add two lists of tensors."""
        return [xx + yy for xx, yy in zip(x, y)]


class TensorFlatten(eqx.Module):
    def __call__(self, x: List[Array]) -> Float[Array, "seq_len dim"]:
        """Flatten and concatenate all signature levels along feature axis."""
        x = [einops.rearrange(xx, "seq ... -> seq (...)") for xx in x]
        x = jnp.concatenate(x, axis=-1)
        return x


def compute_signature(
    x: Float[Array, "seq_len dim"], depth: int = 3, basepoint: float = 0.0, stream=True
) -> List[Array]:
    """Compute iterated signature terms of a path.

    Args:
        x: Input path shaped (seq_len, dim).
        depth: Signature depth to compute.
        basepoint: Optional basepoint padding value.
        stream: If True, return streaming signatures, else final term.

    Returns:
        list[Array]: Signature tensors for each level.
    """

    if basepoint is not None:
        x = jnp.pad(x, ((1, 0), (0, 0)), constant_values=basepoint)

    path_increments = jnp.diff(x, axis=0)
    exp_term = restricted_exp(path_increments[0], depth=depth)

    def f(carry, path_inc):
        ret = mult_fused_restricted_exp(path_inc, carry)
        return ret, ret

    carry, stacked = jax.lax.scan(
        f=f,
        init=exp_term,
        xs=path_increments[1:],
    )

    if stream:
        return [
            jnp.concatenate([first[None, ...], rest], axis=0)
            for first, rest in zip(exp_term, stacked)
        ]

    return carry


def lead_lag(x: Float[Array, "seq_len dim"], n_step_delay=1):
    """Create a lead-lag transform of a stream.

    Args:
        x: Input stream shaped (seq_len, dim).
        n_step_delay: Number of lag steps to include.

    Returns:
        Array: Lead-lag expanded path.
    """

    x_repreated = jnp.repeat(x, n_step_delay + 1, axis=0)
    all = [x_repreated[n_step_delay:]]

    for i in range(n_step_delay - 1, -1, -1):
        all += [x_repreated[i : -(n_step_delay - i)]]

    return jnp.concatenate(all, axis=-1)


class LeadLagSignature(eqx.Module):

    patch_len: int = eqx.field(static=True)
    signature_depth: int = eqx.field(static=True)

    def __init__(self, depth, patch_len):
        self.patch_len = patch_len
        self.signature_depth = depth

    def __call__(self, x: Float[Array, "seq_len dim"]):
        """Compute windowed lead-lag signatures over the path."""
        seq_len, dim = x.shape

        x = jnp.pad(x, ((self.patch_len - 1, 0), (0, 0)), constant_values=0.0)

        index = jnp.arange(seq_len)

        def _f(carry, i):
            patch = jax.lax.dynamic_slice(x, (i, 0), (self.patch_len, dim))
            lead_lag_patch = lead_lag(patch)
            sig = signature(
                lead_lag_patch,
                depth=self.signature_depth,
                stream=False,
                flatten=False,
            )
            return carry, sig

        # output shape: (number_strides, patch_len, dim)
        _, output = jax.lax.scan(f=_f, init=None, xs=index)
        return output


class Signature(eqx.Module):

    depth: int = eqx.field(static = True)
    basepoint: float = eqx.field(static = True)

    def __init__(self, depth: int = 3, basepoint: float = 0.0) -> None:

        self.depth = depth
        self.basepoint = basepoint

    def __call__(self, x: Float[Array, "seq_len dim"]):
        """Compute streamed signatures with an optional basepoint."""

        # add basepoint
        x = jnp.pad(x, ((1, 0), (0, 0)), constant_values=self.basepoint)
        sig = signature(x, depth=self.depth, stream=True, flatten=False)
        return sig
