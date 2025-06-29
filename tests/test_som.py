import numpy as np
from utils.som import SOM

def test_som_output_shape():
    som = SOM(5, 5, seed=1)
    input_data = np.random.rand(5, 3)
    som.train(input_data, 20)
    weights = som.get_weights()
    assert weights.shape == (5, 5, 3)
    assert (weights >= 0).all() and (weights <= 1).all()

def test_som_evaluation_range():
    som = SOM(5, 5, seed=42)
    data = np.random.rand(10, 3)
    som.train(data, 30)
    error = som.evaluate(data)
    assert isinstance(error, float)
    assert error >= 0