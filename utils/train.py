# train.py
import numpy as np
import logging
from som import SOM
from visualize import save_weights_image

def generate_and_train(
    seed: int,
    iterations: int,
    width: int,
    height: int,
    filename: str,
    input_size: int = 10
) -> SOM:
    """
    Generate random input and train a SOM, saving the result to an image file.
    """
    logging.info(f"Training SOM {width}x{height} for {iterations} iterations...")
    rng = np.random.RandomState(seed)
    input_data = rng.rand(input_size, 3)
    som = SOM(width, height, seed=seed)
    som.train(input_data, iterations)
    save_weights_image(som.get_weights(), filename)
    error = som.evaluate(input_data)
    logging.info(f"Average quantization error: {error:.4f}")
    logging.info(f"Saved trained SOM image to {filename}")
    return som