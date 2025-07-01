import numpy as np
from typing import Optional


class SOM:
    """
    Self-Organizing Map (SOM) for unsupervised learning of input vectors.
    """

    def __init__(
        self,
        width: int,
        height: int,
        input_dim: int = 3,
        alpha: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.width = width
        self.height = height
        self.input_dim = input_dim
        self.alpha0 = alpha
        self.sigma0 = max(width, height) / 2

        if seed is not None:
            np.random.seed(seed)
        self.weights = np.random.rand(width, height, input_dim)

    def train(self, input_data: np.ndarray, n_iterations: int, eval_interval: int = 10, patience: int = 5, min_delta: float = 1e-4) -> None:
        """
        Train the SOM using the input data with optional early stopping.

        Args:
            input_data (np.ndarray): Input vectors of shape (n_samples, input_dim).
            n_iterations (int): Maximum number of training iterations.
            eval_interval (int): How often to evaluate quantization error.
            patience (int): Number of evaluation steps with no improvement before stopping.
            min_delta (float): Minimum change in quantization error to count as improvement.
        """
        λ = n_iterations / np.log(self.sigma0)
        x_grid, y_grid = np.meshgrid(
            np.arange(self.width), np.arange(self.height), indexing="ij"
        )

        last_errors = []

        for t in range(n_iterations):
            sigma_t = self.sigma0 * np.exp(-t / λ) # Radius
            alpha_t = self.alpha0 * np.exp(-t / λ) # Learning Rate

            for vt in input_data:
                diff = self.weights - vt
                distances = np.sum(diff ** 2, axis=2)
                bmu_idx = np.unravel_index(np.argmin(distances), (self.width, self.height))

                dx = x_grid - bmu_idx[0]
                dy = y_grid - bmu_idx[1]
                dist_sq = dx ** 2 + dy ** 2
                theta = np.exp(-dist_sq / (2 * sigma_t ** 2))[:, :, np.newaxis]

                self.weights += alpha_t * theta * (vt - self.weights)

            # Early stopping check
            if t % eval_interval == 0:
                err = self.evaluate(input_data)
                last_errors.append(err)
                if len(last_errors) > patience:
                    recent = last_errors[-patience:]
                    if max(np.abs(np.diff(recent))) < min_delta:
                        print(f"Early stopping at iteration {t}, quantization error: {err:.5f}")
                        break

    def get_weights(self) -> np.ndarray:
        return np.clip(self.weights, 0, 1)

    def evaluate(self, input_data: np.ndarray) -> float:
        """
        Evaluate the SOM by computing the average quantization error.

        Args:
            input_data (np.ndarray): Input vectors of shape (n_samples, input_dim).

        Returns:
            float: Average quantization error.
        """
        total_error = 0
        for vt in input_data:
            diff = self.weights - vt
            distances = np.sum(diff ** 2, axis=2)
            total_error += np.sqrt(np.min(distances))
        return total_error / len(input_data)
