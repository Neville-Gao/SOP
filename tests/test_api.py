from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_train_image_api():
    payload = {
        "width": 10,
        "height": 10,
        "iterations": 50,
        "input_size": 5,
        "seed": 123
    }
    response = client.post("/train-image", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0