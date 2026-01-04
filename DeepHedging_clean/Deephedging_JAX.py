import os
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import equinox as eqx
import numpy as np
from SigFormer.model import SigFormer, Config
from SigFormer.utils import exp_utility_loss
from plots.plot import plot_holdings_hedge, plot_train_vs_val_loss


class DeepHedgingJAX:
    def __init__(
        self,
        data_generator,
        in_dim,
        out_dim,
        idx_asset,
        is_plot,
        plot_path,  
        model_dim=8,
        n_heads=2,
        d_ff=32,
        order=3,
        n_blocks=2,
        lr=1e-3,
        seed=0,
        n_regimes=None,
        risk_aversion=1.3,
        clip_X=True,
        save_path=None,
        name="sigformer_weights",
    ):
        """
        Args:
            data_generator: Object exposing build_data/make_test_data and option params.
            in_dim: Number of input features (assets).
            out_dim: Number of hedging instruments (often equals in_dim).
            idx_asset: Index of the underlying to price the payoff on.
            is_plot: Whether to plot training/test outputs.
            plot_path: Destination path for plots.
            model_dim: Transformer hidden dimension.
            n_heads: Number of attention heads.
            d_ff: Feedforward size.
            order: Signature truncation order.
            n_blocks: Number of attention blocks.
            lr: Learning rate.
            seed: PRNG seed.
            n_regimes: Optional regime count if regimes are provided.
            risk_aversion: Exponential utility risk aversion.
            clip_X: Whether to clip terminal wealth for stability.
            save_path: Where to store trained weights.
            name: Base file name for saved weights.
        """

        self.data_generator = data_generator
        self.idx_asset = idx_asset
        self.n_regimes = n_regimes or getattr(self.data_generator, "n_regimes", None)
        self.data_kind = "prices"  # "prices" or "returns"
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.test_returns = None
        self.test_prices = None
        self.risk_aversion = risk_aversion
        self.indim = in_dim
        self.clip_X = clip_X
        self.is_plot = is_plot
        self.plot_path = plot_path
        self.idx_assets_to_hedge = data_generator.idx_assets_to_hedge
    
        
        cfg = Config(
            in_dim=in_dim,
            out_dim=out_dim,
            dim=model_dim,
            num_heads=n_heads,
            d_ff=d_ff,
            dropout=0.0,
            n_attn_blocks=n_blocks,
            order=order,
        )


        key = jrandom.PRNGKey(seed)
        self.key, model_key = jrandom.split(key)

        # model 
        self.model = SigFormer(cfg, key=model_key)

        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))

        self.save_path = save_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "trained_models"
        )
        os.makedirs(self.save_path, exist_ok=True)
        self.model_name = name
        tag = self._data_tag()
        if tag == "bs":
            self.model_file = os.path.join(self.save_path, f"{self.model_name}_{self.indim - self.n_regimes}_assets_for_{tag}.eqx")
        else : 
            self.model_file = os.path.join(self.save_path, f"{self.model_name}_{self.indim}_assets_for_{tag}.eqx")
        print(f"Loading SigFormer weights from: {self.model_file} (exists={os.path.exists(self.model_file)})")

    # ---------------------- Data utilities ----------------------
    def _data_tag(self):
        """
        Identify a short tag for the generator type to name saved weights.

        Returns:
            str: Tag describing the data source.
        """
        gen = self.data_generator.__class__.__name__.lower()
        if "diffusion" in gen:
            return "diffusion"
        if "market" in gen:
            return "marketdata"
        return "bs"
    def _lag(self, x):
        """
        Lag features by one step, padding the first entry with zeros.

        Args:
            x: Array shaped (B, Tm1, F) or (B, Tm1).

        Returns:
            jnp.ndarray: Lagged array with same shape as x.
        """
        x = self._to_jnp(x)
        if x.ndim == 2:  # (B, Tm1) -> (B, Tm1, 1)
            x = x[..., None]
        zero = jnp.zeros_like(x[:, :1, :])
        return jnp.concatenate([zero, x[:, :-1, :]], axis=1)
    def _to_jnp(self, arr):
        """
        Convert numpy/torch-like inputs to a JAX array.

        Args:
            arr: Input array or tensor; None is returned unchanged.

        Returns:
            jnp.ndarray | None: JAX array view of the input.
        """
        if arr is None:
            return None
        # handle torch tensors transparently
        if hasattr(arr, "detach"):
            arr = arr.detach()
        if hasattr(arr, "cpu"):
            arr = arr.cpu()
        return jnp.array(np.array(arr))

    def _to_jnp_tree(self, obj):
        """
        Apply `_to_jnp` across tuples so trees stay JAX friendly.

        Args:
            obj: Single array or tuple of arrays.

        Returns:
            Same structure with JAX arrays.
        """
        if isinstance(obj, tuple):
            return tuple(self._to_jnp_tree(x) for x in obj)
        return self._to_jnp(obj)

    def _detect_data_kind(self):
        """
        Default heuristic to decide whether build_data returns returns or prices.
        Diffusion generators contain 'diffusion' in their class name.

        Returns:
            str: "returns" or "prices".
        """
        name = self.data_generator.__class__.__name__.lower()
        if "diffusion" in name:
            return "returns"
        return "prices"

    def _returns_to_prices(self, x):
        """
        x: scaled log-returns (B, T, N)
        prices: (B, T+1, N)

        Args:
            x: Log-returns shaped (B, T, N) or (B, T).

        Returns:
            jnp.ndarray: Reconstructed prices with leading S0 step.
        """
        x = self._to_jnp(x)
        if x.ndim == 2:
            x = x[..., None]
   
        r = x 

        s0 = getattr(self.data_generator, "S0", 100.0)
        s0_vec = jnp.full((r.shape[0], 1, r.shape[-1]), s0)

        logS = jnp.cumsum(r, axis=1)
        S = s0_vec * jnp.exp(logS)
        return jnp.concatenate([s0_vec, S], axis=1)


    def _prepare_batch(self, batch):
        """
        Normalize batch into (features, prices) for loss computation.
        - If data_kind == 'prices', batch is either prices or (prices, regimes).
        - If data_kind == 'returns', batch is returns; prices are reconstructed.

        Args:
            batch: Array or tuple (data, regimes) depending on generator.

        Returns:
            tuple: (feats, prices) ready for the model/loss.
        """
        regimes = None
        if isinstance(batch, tuple) and len(batch) == 2:
            prices_or_returns, regimes = batch
        else:
            prices_or_returns = batch

        if self.data_kind == "returns":
            returns = self._to_jnp(prices_or_returns)
            prices = self._returns_to_prices(returns)
            feats = self._lag(returns) 
        else:
            prices = self._to_jnp(prices_or_returns)
            eps = 1e-6
            logret = jnp.log((prices[:, 1:, :] + eps) / (prices[:, :-1, :] + eps))  # (B, T-1, N)
            feats = self._lag(logret)    


        if regimes is not None:
            if self.n_regimes is None:
                raise ValueError("n_regimes must be set when regimes are provided.")
            reg_arr = self._to_jnp(regimes)
            if reg_arr.ndim == 2:
                reg_arr = jax.nn.one_hot(reg_arr.astype(int), self.n_regimes)
            # ensure float
            reg_arr = reg_arr.astype(jnp.float32)
            reg_feats = reg_arr[:, :-1, :]  # align with T-1 returns
            feats = jnp.concatenate([feats, reg_feats], axis=-1)

        return feats, prices

    def _split_data(self, data):
        """
        Accepts outputs of build_data which may be:
        - (train, test, val)
        - (train, test)
        - single array (train)

        Args:
            data: Output from data_generator.build_data().

        Returns:
            tuple: (train, test) JAX-ready arrays or tuples.
        """
        self.test_returns = None
        self.test_prices = None
 
        train = data    
        generated_test = self.data_generator.make_test_data()
 
        self.test_prices, self.test_returns = generated_test
        test = self.test_returns

        return (
            self._to_jnp_tree(train),
            self._to_jnp_tree(test) if test is not None else None,
        )

    def get_data(self, data_kind=None):
        """
        Build data from the generator and split test into validation/test.

        Args:
            data_kind: Optional override for "prices" or "returns".
        """
        print("Generating data...")
        self.data_kind = data_kind or self._detect_data_kind()
        train, test, _= self.data_generator.build_data()
        split = int(0.1 * test.shape[0])
        self.train_data = train
        self.val_data   = test[:split]
        self.test_data  = test[split:]
        print("Data generated.")

    def get_data_BS(self, idx_assets_to_hedge, val_ratio=0.2):
        """
        Build Black-Scholes prices/regimes data and create train/val/test splits.

        Args:
            idx_assets_to_hedge: Indices of assets to hedge.
            val_ratio: Fraction of test paths used for validation.
        """
        print("Generating BS data...")
        self.data_kind = "prices"
        self.n_regimes = getattr(self.data_generator, "n_regimes", self.n_regimes)

        prices, regimes = self.data_generator.build_data(idx_assets_to_hedge)
        prices = self._to_jnp_tree(prices)
        regimes = self._to_jnp_tree(regimes)


        self.train_data = (prices, regimes)

        test_prices, test_regimes = self.data_generator.make_test_data(idx_assets_to_hedge)
        N = test_prices.shape[0]
        val_size = int(0.1 * N)

        # Validation split
        val_prices = test_prices[:val_size]
        val_regimes = test_regimes[:val_size]

        # Test split
        test_prices_ = test_prices[val_size:]
        test_regimes_ = test_regimes[val_size:]

        # Store as JAX trees
        self.val_data = (
            self._to_jnp_tree(val_prices),
            self._to_jnp_tree(val_regimes),
        )

        self.test_data = (
            self._to_jnp_tree(test_prices_),
            self._to_jnp_tree(test_regimes_),
        )
        print("Data generated.")


    def get_data_Diffusion(self):
        """
        Build diffusion returns data and create train/val/test splits.

        Returns:
            None
        """
        print("Generating diffusion data...")
        self.data_kind = "returns"
        raw = self.data_generator.build_data()
        train, test = self._split_data(raw)
        split = int(0.1 * test.shape[0])
        self.train_data = train
        self.val_data   = test[:split]
        self.test_data  = test[split:]

        print("Data generated.")

    # loss on a batch
    def _loss(self, model, batch, *, key):
        """
        Compute exponential utility loss on one batch.

        Args:
            model: SigFormer model.
            batch: Features/prices or tuple as prepared by get_data*.
            key: PRNG key.

        Returns:
            jnp.ndarray: Scalar loss.
        """
        feats, prices = self._prepare_batch(batch)
        B = feats.shape[0]
        keys = jrandom.split(key, B)

        def forward_one(ret, k):
            return model(ret, key=k)

        holding = jax.vmap(forward_one)(feats, keys)

        loss = exp_utility_loss(
            holding,
            prices,
            strike=self.data_generator.K,
            idx_asset=self.idx_asset,
            lamb=self.risk_aversion,
            clip_X=self.clip_X,
        )
        return loss

    def _loss_and_pnl(self, model, batch, *, key):
        """
        Return loss plus average raw PnL for monitoring.

        Args:
            model: SigFormer model.
            batch: Features/prices or tuple as prepared by get_data*.
            key: PRNG key.

        Returns:
            tuple: (loss, mean_raw_pnl).
        """
        feats, prices = self._prepare_batch(batch)

        B = feats.shape[0]
        keys = jrandom.split(key, B)

        def forward_one(x, k):
            return model(x, key=k)

        holding = jax.vmap(forward_one)(feats, keys)  # (B, T, Nhedge)

        dS = prices[:, 1:, :] - prices[:, :-1, :]     # (B, T, N)
        pnl = jnp.sum(holding * dS[:, :, :holding.shape[-1]], axis=(1, 2))

        ST = prices[:, -1, self.idx_asset]
        payoff = jnp.maximum(ST - self.data_generator.K, 0.0)
        raw_pnl = pnl - payoff

        X = raw_pnl
        if self.clip_X:
            X = jnp.maximum(X, -10.0)

        loss = (1.0 / self.risk_aversion) * jnp.log(jnp.mean(jnp.exp(-self.risk_aversion * X)))
        name = self.data_generator.__class__.__name__.lower()

        if self.is_plot: 
            plot_holdings_hedge(holding,prices ,name = f'{name} SigFormer', plot_path = self.plot_path, idx_assets = self.data_generator.idx_assets_to_hedge)


        return loss, jnp.mean(raw_pnl)

    def eval_loss(self, model, data, batch_size=256):
        """
        Evaluate mean loss over a dataset in mini-batches.

        Args:
            model: SigFormer model.
            data: Dataset or tuple of arrays.
            batch_size: Mini-batch size.

        Returns:
            float: Mean loss.
        """
        losses = []
        N = data[0].shape[0] if isinstance(data, tuple) else data.shape[0]

        for start in range(0, N, batch_size):
            end = start + batch_size
            if isinstance(data, tuple):
                batch = tuple(arr[start:end] for arr in data)
            else:
                batch = data[start:end]

            self.key, loss_key = jrandom.split(self.key)
            loss = self._loss(model, batch, key=loss_key)
            losses.append(loss)

        return float(jnp.mean(jnp.stack(losses)))



    def train_step(self, model, opt_state, batch, *, key):
        """
        Run one optimizer step.

        Args:
            model: eqx.Module.
            opt_state: Optimizer state.
            batch: (B, T, N_assets) data.
            key: PRNG key.

        Returns:
            tuple: (updated_model, updated_opt_state, loss).
        """

        # use eqx.filter_value_and_grad so grads only on arraysBu
        loss, grads = eqx.filter_value_and_grad(self._loss)(model, batch, key=key)

        updates, opt_state = self.optimizer.update(
            grads, opt_state, params=eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss

    # Full training loop
    def train(self, epochs=50, batch_size=256):
        """
        Train the SigFormer with validation tracking and save the weights.

        Args:
            epochs: Number of epochs.
            batch_size: Mini-batch size.

        Returns:
            None
        """
        if self.train_data is None or self.val_data is None:
            raise ValueError("Call get_data/get_data_BS/get_data_Diffusion before training.")

        print("Training JAX SigFormer with validation...")

        train_data = (
            tuple(self.train_data)
            if isinstance(self.train_data, tuple)
            else jnp.array(self.train_data)
        )
        val_data = (
            tuple(self.val_data)
            if isinstance(self.val_data, tuple)
            else jnp.array(self.val_data)
        )

        N = train_data[0].shape[0] if isinstance(train_data, tuple) else train_data.shape[0]
        num_batches = max(1, (N + batch_size - 1) // batch_size)

        train_val_results = []

        for epoch in range(epochs):
            # Shuffle training data
            self.key, subkey = jrandom.split(self.key)
            perm = jrandom.permutation(subkey, N)
            if isinstance(train_data, tuple):
                shuffled = tuple(arr[perm] for arr in train_data)
            else:
                shuffled = train_data[perm]

            train_losses = []

            for i in range(num_batches):
                start = i * batch_size
                end = (i + 1) * batch_size
                if isinstance(shuffled, tuple):
                    batch = tuple(arr[start:end] for arr in shuffled)
                else:
                    batch = shuffled[start:end]

                self.key, loss_key = jrandom.split(self.key)
                self.model, self.opt_state, loss = self.train_step(
                    self.model, self.opt_state, batch, key=loss_key
                )
                train_losses.append(loss)

            train_loss = float(jnp.mean(jnp.stack(train_losses)))
            val_loss = self.eval_loss(self.model, val_data, batch_size=batch_size)

            train_val_results.append((train_loss, val_loss))

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"train loss = {train_loss:.6f} | "
                f"val loss = {val_loss:.6f}"
            )

        # Save model
        eqx.tree_serialise_leaves(self.model_file, self.model)
        print(f"Model saved to {self.model_file}")

        if self.is_plot:
            plot_train_vs_val_loss(
                train_val_results,
                plot_path=self.plot_path,
                model_name=self.model_name,
            )

        print("JAX training finished.")


    # Testing loop
    def test(self, batch_size=None):
        """
        Load saved weights (if any) and evaluate on the test split.

        Args:
            batch_size: Optional override for batch size.

        Returns:
            tuple: (loss, mean_pnl).
        """
        if self.test_data is None:
            raise ValueError("Call get_data/get_data_BS/get_data_Diffusion before testing.")

        print("Testing JAX SigFormer...")
        if os.path.exists(self.model_file):
            self.model = eqx.tree_deserialise_leaves(self.model_file, self.model)
        test_data = (
            tuple(self.test_data)
            if isinstance(self.test_data, tuple)
            else jnp.array(self.test_data)
        )
        losses = []
        pnls = []
        if self.data_kind == "prices" :
            if batch_size is None:
                batch_size = test_data[0].shape[0] if isinstance(test_data, tuple) else test_data.shape[0]

            total = test_data[0].shape[0] if isinstance(test_data, tuple) else test_data.shape[0]
            for start in range(0, total, batch_size):
                end = start + batch_size
                if isinstance(test_data, tuple):
                    batch = tuple(arr[start:end] for arr in test_data)
                else:
                    batch = test_data[start:end]
                self.key, loss_key = jrandom.split(self.key)
                loss, pnl = self._loss_and_pnl(self.model, batch, key=loss_key)
                losses.append(loss)
                pnls.append(pnl)
        else:
            test_returns = jnp.array(self.test_data) 
            if batch_size is None:
                batch_size = test_returns.shape[0]

            total = test_returns.shape[0]
            for start in range(0, total, batch_size):
                end = start + batch_size
                batch = test_returns[start:end]      
                self.key, loss_key = jrandom.split(self.key)
                loss, pnl = self._loss_and_pnl(self.model, batch, key=loss_key)
                losses.append(loss)
                pnls.append(pnl)


        loss = float(jnp.mean(jnp.stack(losses)))
        mean_pnl = float(jnp.mean(jnp.stack(pnls))) if pnls else 0.0
        
        print("Test loss:", loss)
        print("Test mean PnL:", mean_pnl)
        return loss, mean_pnl

   
    def train_MarketData(self, epochs=50, batch_size=256):
        """
        Convenience wrapper to train on market price data.

        Args:
            epochs: Number of epochs.
            batch_size: Mini-batch size.
        """
        self.data_kind = "prices"
        return self.train(epochs=epochs, batch_size=batch_size)

    def train_BS(self, epochs=50, batch_size=256):
        """
        Convenience wrapper to train on Black-Scholes price data.

        Args:
            epochs: Number of epochs.
            batch_size: Mini-batch size.
        """
        self.data_kind = "prices"
        return self.train(epochs=epochs, batch_size=batch_size)

    def train_Diffusion(self, epochs=50, batch_size=256):
        """
        Convenience wrapper to train on diffusion return data.

        Args:
            epochs: Number of epochs.
            batch_size: Mini-batch size.
        """
        self.data_kind = "returns"
        return self.train(epochs=epochs, batch_size=batch_size)
