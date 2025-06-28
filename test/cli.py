import argparse
import logging
from utils.train import generate_and_train
from utils.som import SOM
import numpy as np


def _run_tests():
    def test_som_output_shape():
        som = SOM(5, 5, seed=1)
        dummy_input = np.random.rand(5, 3)
        som.train(dummy_input, 10)
        weights = som.get_weights()
        assert weights.shape == (5, 5, 3), "Incorrect weight shape"
        assert (weights >= 0).all() and (weights <= 1).all(), "Weights out of bounds"

    logging.info("Running unit tests...")
    test_som_output_shape()
    logging.info("All tests passed.")


def main():
    parser = argparse.ArgumentParser(description="Train a Self-Organizing Map (SOM)")
    parser.add_argument("--width", type=int, default=10, help="SOM grid width")
    parser.add_argument("--height", type=int, default=10, help="SOM grid height")
    parser.add_argument("--iterations", type=int, default=100, help="Training iterations")
    parser.add_argument("--output", type=str, default="som.png", help="Output image filename")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--input-size", type=int, default=10, help="Number of random input vectors")
    parser.add_argument("--test", action="store_true", help="Run internal unit tests")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.test:
        _run_tests()
    else:
        generate_and_train(
            seed=args.seed,
            iterations=args.iterations,
            width=args.width,
            height=args.height,
            filename=args.output,
            input_size=args.input_size
        )


if __name__ == "__main__":
    main()