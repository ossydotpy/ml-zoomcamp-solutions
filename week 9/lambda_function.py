import numpy as np
import tflite_runtime.interpreter as tflite

from io import BytesIO
from urllib import request
from PIL import Image



def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img
    

def process_input(x):
    x = np.array(x, dtype='float32')
    X = 1./255 * np.array([x])
    return X



def predict(url):

    image = download_image(url)

    x = prepare_image(image, (150,150))
    X = process_input(x)

    interpreter = tflite.Interpreter(model_path='bees-wasps-v2.tflite')
    interpreter.allocate_tensors()

    input_tensor_index = interpreter.get_input_details()[0]['index']
    output_tensor_index = interpreter.get_output_details()[0]['index']

    interpreter.set_tensor(input_tensor_index, X)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_tensor_index)[0].tolist()

    return predictions



def lambda_handler(event, context):
    url = event['url']
    prediction = predict(url)

    return prediction
