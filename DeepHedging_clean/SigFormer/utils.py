import jax.random as jrandom
import jax.numpy as jnp

def split_key(key):
    """Split a PRNG key once, safely handling None."""
    return None if key is None else jrandom.split(key, 1)[0]


def exp_utility_loss(holding, prices, strike, lamb=1.3, idx_asset=0,
                     clip_X=True):
    """Compute exponential utility loss from hedging PnL.

    Args:
        holding: Hedge positions over time (B, T, H).
        prices: Asset prices (B, T+1, N).
        strike: Option strike.
        lamb: Risk aversion parameter.
        idx_asset: Underlying asset index for payoff.
        clip_X: Whether to cap losses for stability.

    Returns:
        jnp.ndarray: Scalar loss value.
    """

    dS = prices[:, 1:, :] - prices[:, :-1, :]
    pnl = jnp.sum(holding * dS[:, :, :holding.shape[-1]], axis=(1, 2))

    ST = prices[:, -1, idx_asset]
    payoff = jnp.maximum(ST - strike, 0.0)
    X = pnl - payoff
    if clip_X:
        X = jnp.maximum(X, -10.0)
    loss = (1.0 / lamb) * jnp.log(jnp.mean(jnp.exp(-lamb * X)))

    return loss
