import matplotlib.pyplot as plt
import numpy as np

def save_weights_image(weights: np.ndarray, filename: str) -> None:
    plt.imshow(weights)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()