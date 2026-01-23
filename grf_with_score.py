"""
Gaussian Random Field with efficient score computation via FFT.

Adapted from: https://github.com/devitocodes/devito/blob/master/examples/cfd/09_Darcy_flow_equation.ipynb
"""

import math
import numpy as np
import numpy.fft as fft

__all__ = ["GaussianRandomField"]


class GaussianRandomField:
    """
    Class for generating realizations of a Gaussian random field.

    Parameters:
    -----------
    dim : int
        Dimensionality of the random field (e.g., 2 for 2D fields).
    size : int
        Size of the grid for the random field in each dimension.
    alpha : float, optional
        Power exponent that controls the smoothness of the field. Defaults to 2.
    tau : float, optional
        Scale parameter that influences the correlation length of the field.
        Defaults to 4.
    sigma : float, optional
        Internal scaling parameter. If None, it is automatically calculated
        based on 'alpha', 'tau', and 'dim'. Defaults to None.
    boundary : str, optional
        Type of boundary conditions for the field. Currently, only "periodic" is
        supported. Defaults to "periodic".
    bounds : tuple, optional
        (low, high) bounds for output values. If provided, the GRF is linearly
        scaled so that ~99.7% of values fall within this range (±3σ).
        mean = (low + high) / 2, std = (high - low) / 6.

    Example:
    --------
    # GRF with values mostly in [0.01, 0.5]
    grf = GaussianRandomField(2, size=128, bounds=(0.01, 0.5))
    samples = grf.sample(10)
    """

    def __init__(
        self,
        dim: int,
        size: int,
        alpha: float = 4,
        tau: float = 5,
        sigma: float = None,
        boundary: str = "periodic",
        bounds: tuple = None,
    ) -> None:
        self.dim = dim
        self.size = size
        self.alpha = alpha
        self.tau = tau

        # Compute mean and std from bounds if provided
        if bounds is not None:
            low, high = bounds
            self.mean = (low + high) / 2
            self.target_std = (high - low) / 6  # ±3σ covers the range
        else:
            self.mean = 0.0
            self.target_std = None

        if dim != 2:
            raise ValueError("Only 2D fields are supported.")

        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - self.dim))
        self.sigma = sigma

        # Wavenumbers
        k_max = size // 2
        k = np.concatenate([np.arange(0, k_max), np.arange(-k_max, 0)])
        k_x, k_y = np.meshgrid(k, k, indexing="ij")

        # Power spectrum: λ(k) ∝ (4π²|k|² + τ²)^(-α)
        spectrum = (4 * math.pi**2 * (k_x**2 + k_y**2) + tau**2) ** (-alpha / 2)

        # sqrt(eigenvalues) for sampling (zero mean)
        self.sqrt_eig = size**2 * math.sqrt(2) * sigma * spectrum
        self.sqrt_eig[0, 0] = 0  # zero mean in Fourier space

        # Eigenvalues
        self.eig = self.sqrt_eig**2

        # Compute natural GRF std
        self.natural_std = np.sqrt(np.sum(self.eig) / size**4)

        # Scaling factor for output
        if self.target_std is not None:
            self.std_scale = self.target_std / self.natural_std
        else:
            self.std_scale = 1.0
            self.target_std = self.natural_std  # for consistency

        # Scaled eigenvalues for the output distribution: eig_scaled = std_scale² * eig
        self.eig_scaled = self.std_scale**2 * self.eig

        # Inverse of scaled eigenvalues for score computation
        # Use Tikhonov regularization to handle small eigenvalues
        # This ensures high-frequency modes are penalized (pushed to zero)
        # rather than ignored (which leaves them as noise)
        eig_reg = 1e-6 * np.max(self.eig_scaled)
        self.inv_eig_scaled = 1.0 / (self.eig_scaled + eig_reg)

    def sample(self, N: int) -> np.ndarray:
        """
        Generate N samples from the Gaussian random field.

        Returns array of shape (N, size, size) with values scaled to target bounds.
        """
        # Generate random coefficients in Fourier space
        coeff = np.random.randn(N, self.size, self.size)
        coeff = self.sqrt_eig * coeff

        # Inverse FFT to get natural GRF (zero mean, natural_std)
        field = fft.ifftn(coeff, axes=(-2, -1)).real

        # Linear transform: scale to target_std and shift to mean
        return field * self.std_scale + self.mean

    def score(self, M: np.ndarray) -> np.ndarray:
        """
        Compute score: ∇_M log p(M) = -K_scaled⁻¹ (M - mean)

        Where K_scaled = std_scale² * K is the covariance of the scaled GRF.
        """
        single = M.ndim == 2
        if single:
            M = M[np.newaxis]

        # Center the input
        M_centered = M - self.mean

        # Score: -K_scaled⁻¹ (M - mean) via FFT
        M_fft = fft.fftn(M_centered, axes=(-2, -1))
        score = -fft.ifftn(self.inv_eig_scaled * M_fft, axes=(-2, -1)).real

        return score[0] if single else score

    def log_prob(self, M: np.ndarray) -> float:
        """
        Compute log p(M) = -0.5 (M-μ)ᵀ K_scaled⁻¹ (M-μ) + const
        """
        M_centered = M - self.mean

        M_fft = fft.fftn(M_centered)
        n_elements = M.size
        return -0.5 * np.sum(np.abs(M_fft) ** 2 * self.inv_eig_scaled).real / n_elements


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(42)

    # Create GRF with bounds to have values mostly in [0.01, 0.5]
    grf = GaussianRandomField(2, size=128, bounds=(0.01, 0.5))
    print(f"GRF: mean={grf.mean}, target_std={grf.target_std:.4f}, natural_std={grf.natural_std:.4f}, std_scale={grf.std_scale:.6f}")
    print(f"max(inv_eig_scaled)={np.max(grf.inv_eig_scaled):.2e}, max(eig_scaled)={np.max(grf.eig_scaled):.2e}")
    samples = grf.sample(10)
    print(f"Samples: min={samples.min():.4f}, max={samples.max():.4f}, has_nan={np.any(np.isnan(samples))}")

    # Test score on a sample
    test_score = grf.score(samples[0])
    print(f"Score of sample: min={test_score.min():.4f}, max={test_score.max():.4f}, has_nan={np.any(np.isnan(test_score))}")

    # Plot samples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(samples[i], cmap="viridis", origin="lower")
        ax.set_title(f"Sample {i + 1}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle(f"GRF Samples (α={grf.alpha}, τ={grf.tau}, mean={grf.mean:.3f}, std={grf.target_std:.3f})", fontsize=14)
    plt.tight_layout()
    plt.savefig("grf_samples.png", dpi=150, bbox_inches="tight")

    # Score visualization
    score = grf.score(samples[0])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(samples[0], cmap="viridis", origin="lower")
    axes[0].set_title("Sample M")
    axes[0].axis("off")
    axes[1].imshow(score, cmap="RdBu_r", origin="lower")
    axes[1].set_title(r"Score $\nabla_M \log p(M)$")
    axes[1].axis("off")
    axes[2].imshow(np.abs(score), cmap="hot", origin="lower")
    axes[2].set_title("|Score|")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig("grf_score.png", dpi=150, bbox_inches="tight")

    print(f"Samples: {samples.shape}, Score: {score.shape}")

    # --- Gradient descent experiment ---
    # Start from random noise and optimize to maximize log_prob
    num_iterations = 5000
    learning_rate = 0.01  # small lr since scores can be large for tight priors

    # Sample from prior (for comparison)
    sample_from_prior = grf.sample(1)[0]
    print(
        f"Log prob of sample from prior: {grf.log_prob(sample_from_prior):.4f}"
    )

    # Start from noisy values around the mean
    u_init = np.random.randn(grf.size, grf.size)
    u = u_init.copy()
    print(f"Log prob of initial: {grf.log_prob(u):.4f}")

    # Gradient ascent to maximize log_prob
    log_probs = []
    grad_norms = []

    for i in range(num_iterations):
        log_prob = grf.log_prob(u)
        score = grf.score(u)

        log_probs.append(log_prob)
        grad_norms.append(np.linalg.norm(score))

        # Gradient ascent step
        u = u + learning_rate * score

        if (i + 1) % 100 == 0:
            print(
                f"Iteration {i + 1}/{num_iterations}, "
                f"Log prob: {log_prob:.4f}, "
                f"Grad norm: {grad_norms[-1]:.4f}"
            )

    final_log_prob = grf.log_prob(u)
    print(f"\nFinal log prob: {final_log_prob:.4f}")

    # Plot gradient descent results
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].plot(log_probs)
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Log prob")
    axes[0, 0].set_title("Log prob over iterations")

    axes[0, 1].plot(grad_norms)
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Gradient norm")
    axes[0, 1].set_title("Gradient norm over iterations")

    im0 = axes[0, 2].imshow(u_init, cmap="viridis", origin="lower")
    axes[0, 2].set_title("Initial (near mean)")
    axes[0, 2].axis("off")
    plt.colorbar(im0, ax=axes[0, 2])

    im1 = axes[1, 0].imshow(u, cmap="viridis", origin="lower")
    axes[1, 0].set_title(f"Optimized (log_prob={final_log_prob:.2f})")
    axes[1, 0].axis("off")
    plt.colorbar(im1, ax=axes[1, 0])

    im2 = axes[1, 1].imshow(sample_from_prior, cmap="viridis", origin="lower")
    axes[1, 1].set_title(
        f"Sample from prior (log_prob={grf.log_prob(sample_from_prior):.2f})"
    )
    axes[1, 1].axis("off")
    plt.colorbar(im2, ax=axes[1, 1])

    final_score = grf.score(u)
    im3 = axes[1, 2].imshow(final_score, cmap="RdBu_r", origin="lower")
    axes[1, 2].set_title(
        f"Score of optimized (norm={np.linalg.norm(final_score):.4f})"
    )
    axes[1, 2].axis("off")
    plt.colorbar(im3, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig("grf_gradient_descent.png", dpi=150, bbox_inches="tight")
    plt.show()
