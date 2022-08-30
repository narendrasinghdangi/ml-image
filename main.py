from fastapi import FastAPI, File, UploadFile
from keras_preprocessing.sequence import pad_sequences
import uvicorn
import numpy as np
import tensorflow as tf
import cv2
from io import BytesIO
from PIL import Image
from keras.models import load_model

app = FastAPI()

vocab = np.load('vocab.npy', allow_pickle=True)
vocab = vocab.item()
inv_vocab = {v:k for k,v in vocab.items()}


model = load_model("model.h5")
mymodel=load_model('mymodel2.h5')

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.get("/")
async def lol():
    return "go to docs"

@app.get("/ping")
async def ping():
    return "Hello, I think you are alive"


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    MAX_LEN=34
    im=0
    img = read_file_as_image(await file.read())
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (224,224))
    im = np.reshape(im, (1,224,224,3))
    test_feature =mymodel.predict(im).reshape(1,2048)
  
    text_input=["startofseq"]
    count=0
    caption=""
    while count<25:
        count+=1
        encoded=[]
        for i in text_input:
            encoded.append(vocab[i])
        encoded=[encoded]
        encoded=pad_sequences(encoded, padding="post", truncating="post", maxlen=MAX_LEN)
        prediction=np.argmax(model.predict([test_feature, encoded]))
        sampled_word=inv_vocab[prediction]
        caption=caption+" "+sampled_word
        if sampled_word=="endofseq":
            break
        text_input.append(sampled_word)

    return caption[:-9]


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
