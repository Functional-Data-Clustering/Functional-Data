import numpy as np
import os
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


# --------------------------------------------------------------------------- #
class DatasetGenerator:
    """
    Generate, save, load, and plot synthetic time-series datasets that live on
    (or are projected to) low-dimensional manifolds (hypersphere, hyperbolic,
    Swiss roll, Lorenz, pendulum).

    Major improvements vs. the baseline implementation:
    - **Much faster** hypersphere/hyperbolic/swiss-roll generation using
      vectorized NumPy ops (no per-time-step inner loops).
    - **Deterministic RNG** via ``numpy.random.Generator`` (reproducibility).
    - **Lower memory option** using ``dtype`` (float32 by default).
    - **Bug fix**: plotting no longer *regenerates* a new dataset; you can pass
      ``X, y`` so the same data are plotted.
    - **Safer normalizations** and defensive checks.

    Parameters
    ----------
    n_samples : int
        Number of time-series samples to generate.
    n_features : int
        Dimensionality of each sample at each time step.
    n_steps : int
        Length of each time-series (number of time steps).
    n_clusters : int, default=2
        Number of distinct clusters or underlying dynamics.
    base_noise : float, default=0.0
        Base noise level for cluster-dependent perturbations.
    omega_delta : float, default=2.0
        Angular velocity separation multiplier for pendulum dynamics.
    seed : Optional[int]
        Random seed for reproducibility (passed to ``np.random.default_rng``).
    dtype : np.dtype, default=np.float32
        Floating dtype for generated arrays.
    """

    def __init__(
        self,
        n_samples: int,
        n_features: int,
        n_steps: int,
        n_clusters: int = 2,
        base_noise: float = 0.0,
        omega_delta: float = 2.0,
        seed: Optional[int] = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        self.n_samples = int(n_samples)
        self.n_features = int(n_features)
        self.n_steps = int(n_steps)
        self.n_clusters = int(n_clusters)
        self.base_noise = float(base_noise)
        self.omega_delta = float(omega_delta)
        self.rng = np.random.default_rng(seed)
        self.dtype = dtype

    # --------------------------- helpers ----------------------------------- #
    def _cluster_idx(self, i: int) -> int:
        """Map sample index ``i`` to a cluster label in ``[0, n_clusters-1]``.
        Partitions indices approximately equally across clusters.
        """
        idx = int(i * self.n_clusters / self.n_samples)
        return min(idx, self.n_clusters - 1)

    def _rand_unit(self, d: int) -> np.ndarray:
        """Unit vector ~ N(0, I) / ||.|| (shape (d,))."""
        v = self.rng.standard_normal(d).astype(self.dtype, copy=False)
        n = np.linalg.norm(v)
        if n == 0:
            v[0] = 1.0
            return v
        return (v / n).astype(self.dtype, copy=False)

    def _rand_unit_orth(self, a: np.ndarray) -> np.ndarray:
        """Unit vector orthogonal to ``a`` (same shape)."""
        temp = self.rng.standard_normal(a.shape[0]).astype(self.dtype, copy=False)
        b = temp - a * np.dot(a, temp)
        n = np.linalg.norm(b)
        if n == 0:
            # If random temp accidentally colinear, rotate a tiny bit
            b = np.zeros_like(a)
            b[(np.argmax(np.abs(a)) + 1) % a.size] = 1.0
            b = b - a * np.dot(a, b)
            n = np.linalg.norm(b)
        return (b / n).astype(self.dtype, copy=False)

    # --------------------------- generators -------------------------------- #
    def generate_hypersphere(self) -> Tuple[np.ndarray, np.ndarray]:
        """Trajectories on the unit hypersphere, cluster-specific angular speeds.

        Returns
        -------
        X : (n_samples, n_features, n_steps)
        y : (n_samples,)
        """
        X = np.zeros((self.n_samples, self.n_features, self.n_steps), dtype=self.dtype)
        y = np.zeros(self.n_samples, dtype=np.int32)

        t = np.arange(self.n_steps, dtype=self.dtype)
        for i in range(self.n_samples):
            k = self._cluster_idx(i)
            y[i] = k
            a = self._rand_unit(self.n_features)
            b = self._rand_unit_orth(a)
            omega = (k + 1) * (2.0 * np.pi / self.n_steps)
            theta = omega * t
            ct, st = np.cos(theta, dtype=self.dtype), np.sin(theta, dtype=self.dtype)

            # Vectorized trajectory: a*cos + b*sin
            points = a[:, None] * ct[None, :] + b[:, None] * st[None, :]

            noise = self.base_noise * k
            if noise > 0:
                points = points + noise * self.rng.standard_normal(points.shape).astype(self.dtype)
                norms = np.linalg.norm(points, axis=0, keepdims=True)
                # avoid divide-by-zero
                norms = np.where(norms == 0, 1.0, norms)
                points = points / norms

            X[i] = points
        return X, y

    def generate_hyperbolic(self) -> Tuple[np.ndarray, np.ndarray]:
        """Trajectories in the Poincaré ball model of hyperbolic space.

        Returns
        -------
        X : (n_samples, n_features, n_steps)
        y : (n_samples,)
        """
        X = np.zeros((self.n_samples, self.n_features, self.n_steps), dtype=self.dtype)
        y = np.zeros(self.n_samples, dtype=np.int32)

        T_vals = np.linspace(1.0, 2.0, self.n_clusters, dtype=self.dtype)
        t_idx = np.linspace(0.0, 1.0, self.n_steps, dtype=self.dtype)

        for i in range(self.n_samples):
            k = self._cluster_idx(i)
            y[i] = k
            T = T_vals[k]
            v = self._rand_unit(self.n_features)  # direction

            times = T * t_idx
            cosh_t = np.cosh(times).astype(self.dtype, copy=False)
            sinh_t = np.sinh(times).astype(self.dtype, copy=False)

            Xspatial = v[:, None] * sinh_t[None, :]
            denom = (1.0 + cosh_t)[None, :]
            points = Xspatial / denom

            # Clamp to inside of unit ball just in case of numeric drift
            norms = np.linalg.norm(points, axis=0)
            over = norms >= 1.0
            if np.any(over):
                points[:, over] *= (0.999 / norms[over])[None, :]

            X[i] = points
        return X, y

    def generate_swiss_roll(self) -> Tuple[np.ndarray, np.ndarray]:
        """Trajectories on a 3D Swiss-roll manifold, with cluster-varying height.
        If ``n_features < 3``, the first ``n_features`` coordinates are kept.
        """
        X = np.zeros((self.n_samples, self.n_features, self.n_steps), dtype=self.dtype)
        y = np.zeros(self.n_samples, dtype=np.int32)

        h_bins = np.linspace(0.0, 10.0, self.n_clusters + 1, dtype=self.dtype)
        base_t0 = self.rng.uniform(1.5 * np.pi, 3.0 * np.pi, size=self.n_samples).astype(self.dtype)
        base_t1 = base_t0 + (1.5 * np.pi)
        t_lin = np.linspace(0.0, 1.0, self.n_steps, dtype=self.dtype)

        for i in range(self.n_samples):
            k = self._cluster_idx(i)
            y[i] = k
            h_low, h_high = h_bins[k], h_bins[k + 1]
            h = self.rng.uniform(h_low, h_high)

            t_vals = base_t0[i] + (base_t1[i] - base_t0[i]) * t_lin
            x_coord = t_vals * np.cos(t_vals)
            z_coord = t_vals * np.sin(t_vals)
            y_coord = np.full_like(t_vals, h)

            point3 = np.stack([x_coord, y_coord, z_coord], axis=0).astype(self.dtype)
            X[i] = point3[: self.n_features]
        return X, y

    def generate_lorenz(self) -> Tuple[np.ndarray, np.ndarray]:
        """Lorenz attractor trajectories with cluster-specific ``rho``.
        Uses simple forward Euler integration (sequential by nature).
        """
        X = np.zeros((self.n_samples, self.n_features, self.n_steps), dtype=self.dtype)
        y = np.zeros(self.n_samples, dtype=np.int32)

        sigma = self.dtype(10.0)
        beta = self.dtype(8.0 / 3.0)
        dt = self.dtype(0.01)
        rho_vals = np.linspace(14.0, 28.0, self.n_clusters, dtype=self.dtype)

        for i in range(self.n_samples):
            k = self._cluster_idx(i)
            y[i] = k
            rho = rho_vals[k]
            state = self.rng.uniform(-10, 10, size=3).astype(self.dtype)

            for t in range(self.n_steps):
                # record
                if self.n_features <= 3:
                    X[i, : self.n_features, t] = state[: self.n_features]
                else:
                    X[i, :3, t] = state
                    # higher dims zero-padded
                x_s, y_s, z = state
                dx = sigma * (y_s - x_s)
                dy = x_s * (rho - z) - y_s
                dz = x_s * y_s - beta * z
                state = state + dt * np.array([dx, dy, dz], dtype=self.dtype)
        return X, y

    def generate_pendulum(self) -> Tuple[np.ndarray, np.ndarray]:
        """Simple pendulum with cluster-specific angular velocity ranges."""
        g = self.dtype(9.81)
        L = self.dtype(1.0)
        dt = self.dtype(0.02)

        X = np.zeros((self.n_samples, self.n_features, self.n_steps), dtype=self.dtype)
        y = np.zeros(self.n_samples, dtype=np.int32)

        omega_sep = self.dtype(self.omega_delta)
        for i in range(self.n_samples):
            k = self._cluster_idx(i)
            y[i] = k
            theta = self.rng.uniform(-0.5, 0.5)
            omega_low = -0.5 + k * omega_sep
            omega_high = 0.5 + k * omega_sep
            omega = self.rng.uniform(omega_low, omega_high)

            for t in range(self.n_steps):
                features = np.array([np.cos(theta), np.sin(theta), omega], dtype=self.dtype)
                if self.n_features <= features.size:
                    X[i, :, t] = features[: self.n_features]
                else:
                    X[i, : features.size, t] = features
                    # remaining dims are zeros
                # update dynamics
                alpha = -(g / L) * np.sin(theta)
                theta = theta + omega * dt
                omega = omega + alpha * dt
                # wrap angle
                if theta > np.pi:
                    theta -= 2 * np.pi
                elif theta < -np.pi:
                    theta += 2 * np.pi
        return X, y

    # ------------------------------- IO ------------------------------------ #
    @staticmethod
    def save_dataset(X: np.ndarray, y: np.ndarray, filepath: str) -> None:
        """Flatten and save dataset (X, y) to CSV using NumPy (faster than pandas
        for pure numeric dumps). Columns are ``f{feat}_t{step}`` plus a final
        ``label`` column.
        """
        n_samples, n_features, n_steps = X.shape
        flat_X = X.reshape(n_samples, n_features * n_steps)

        header_cols = [f"f{feat}_t{step}" for feat in range(n_features) for step in range(n_steps)]
        header = ",".join(header_cols + ["label"])
        arr = np.concatenate([flat_X, y.reshape(-1, 1)], axis=1)
        np.savetxt(filepath, arr, delimiter=",", header=header, comments="")

    @staticmethod
    def load_dataset(filepath: str, n_features: int, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset CSV (written by :meth:`save_dataset`) back into ``X`` and
        ``y``.
        """
        import pandas as pd  # local import to keep base import time low

        df = pd.read_csv(filepath)
        y = df["label"].astype(np.int32).to_numpy()
        flat_X = df.drop(columns=["label"]).to_numpy()
        n_samples = flat_X.shape[0]
        X = flat_X.reshape(n_samples, n_features, n_steps)
        return X, y

    # ------------------------------ plotting ------------------------------- #
    def plot_dataset(
        self,
        name: Optional[str] = None,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        max_traj: int = 200,
    ) -> None:
        """Plot example trajectories. If ``X, y`` are not provided, ``name`` must
        be provided and the corresponding generator is called.

        ``max_traj`` limits plotted trajectories per figure for speed.
        """
        if X is None or y is None:
            if name is None:
                raise ValueError("Provide either (X, y) or name to generate.")
            if not hasattr(self, f"generate_{name}"):
                raise ValueError(f"No generator named {name}")
            X, y = getattr(self, f"generate_{name}")()

        n_samples, n_feats, n_steps = X.shape
        # Subsample for plotting if needed
        if n_samples > max_traj:
            idx = self.rng.choice(n_samples, size=max_traj, replace=False)
            X = X[idx]
            y = y[idx]
            n_samples = X.shape[0]

        labels = np.unique(y)
        cmap = plt.get_cmap("tab10")
        handles = [mlines.Line2D([], [], color=cmap(int(lbl) % 10), lw=2) for lbl in labels]
        label_names = [f"{lbl}" for lbl in labels]

        if n_feats == 1:
            plt.figure(figsize=(8, 4))
            t = np.arange(n_steps)
            for i in range(n_samples):
                plt.plot(t, X[i, 0], color=cmap(y[i] % 10), alpha=0.9)
            plt.xlabel("Time step")
            plt.ylabel("Feature value")
            plt.legend(handles, label_names)
            plt.title(f"{name or 'dataset'} (1D)")
            plt.tight_layout()
            plt.show()
        elif n_feats == 2:
            fig = plt.figure(figsize=(12, 8))
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3, projection="3d")
            t = np.arange(n_steps)
            for i in range(n_samples):
                ax1.plot(t, X[i, 0], color=cmap(y[i] % 10), alpha=0.9)
                ax2.plot(t, X[i, 1], color=cmap(y[i] % 10), alpha=0.9)
                ax3.plot(X[i, 0], X[i, 1], t, color=cmap(y[i] % 10), alpha=0.9)
            ax1.set_title(f"{name or 'dataset'} - Feature 0 vs Time")
            ax1.set_xlabel("Time step")
            ax1.set_ylabel("Value")
            ax1.legend(handles, label_names)

            ax2.set_title(f"{name or 'dataset'} - Feature 1 vs Time")
            ax2.set_xlabel("Time step")
            ax2.set_ylabel("Value")
            ax2.legend(handles, label_names)

            ax3.set_title(f"{name or 'dataset'} - 3D Trajectories (2 features + time)")
            ax3.set_xlabel("Feature 0")
            ax3.set_ylabel("Feature 1")
            ax3.set_zlabel("Time step")
            ax3.legend(handles, label_names)

            plt.tight_layout()
            plt.show()
        else:
            n_cols = 2
            n_rows = int(np.ceil(n_feats / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
            axes = axes.flatten()
            t = np.arange(n_steps)
            for f in range(n_feats):
                ax = axes[f]
                for i in range(n_samples):
                    ax.plot(t, X[i, f], color=cmap(y[i] % 10), alpha=0.9)
                ax.set_title(f"Feature {f}")
                ax.set_xlabel("Time step")
                ax.set_ylabel(f"$f_{f}(t)$")
                ax.legend(handles, label_names)
            for ax in axes[n_feats:]:
                fig.delaxes(ax)
            fig.suptitle(name or "dataset")
            plt.tight_layout()
            plt.show()


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    output_dir = os.path.join(os.getcwd(), "datasets")
    os.makedirs(output_dir, exist_ok=True)

    # specs: (dataset_name, n_samples, n_features, n_steps, n_clusters)
    specs = [
        ("hypersphere", 100, 3, 100, 2),
        ("hyperbolic", 200, 2, 50, 2),
        ("swiss_roll", 300, 2, 200, 4),
        ("lorenz", 100, 3, 100, 3),
        ("pendulum", 200, 2, 100, 4),
    ]

    for name, n_samples, n_features, n_steps, n_clusters in specs:
        print(
            f"Generating {name}: (n={n_samples}, d={n_features}, T={n_steps}, {n_clusters} clusters)"
        )
        gen = DatasetGenerator(n_samples, n_features, n_steps, n_clusters, base_noise=0.02, seed=0)
        X, y = getattr(gen, f"generate_{name}")()
        # Plot the SAME data we just generated (no hidden regeneration)
        gen.plot_dataset(name=name, X=X, y=y, max_traj=150)
        # Save if desired
        # fp = os.path.join(output_dir, f"{name}.csv")
        # DatasetGenerator.save_dataset(X, y, fp)
