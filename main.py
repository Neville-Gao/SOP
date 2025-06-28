from fastapi import FastAPI
from pydantic import BaseModel
from utils.train import generate_and_train
from utils.visualize import save_weights_image
import base64
from io import BytesIO
import matplotlib.pyplot as plt

app = FastAPI()

class SOMRequest(BaseModel):
    width: int
    height: int
    iterations: int
    input_size: int = 10
    seed: int = 42

@app.post("/train")
def train_som(request: SOMRequest):
    som = generate_and_train(
        seed=request.seed,
        iterations=request.iterations,
        width=request.width,
        height=request.height,
        filename="temp_output.png",
        input_size=request.input_size
    )
    # Convert image to base64
    buf = BytesIO()
    plt.imshow(som.get_weights())
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return {"image_base64": img_base64}
