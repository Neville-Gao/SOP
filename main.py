#%%
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from utils.train import generate_and_train
from io import BytesIO
import matplotlib.pyplot as plt

#%%
app = FastAPI()

class SOMRequest(BaseModel):
    width: int
    height: int
    iterations: int
    input_size: int = 10
    seed: int = 42

#%%
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

    # Export image
    buf = BytesIO() # BytesIO() is a class that lets you create an in-memory binary stream
    plt.imshow(som.get_weights())
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0) # resets the file pointer (cursor) of the BytesIO back to the start of the stream
    return StreamingResponse(buf, media_type="image/png")

# %%
