import os
import flask 
from flask import Flask, redirect, url_for, request, render_template
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import io

app = flask.Flask(__name__)
model = None
MODEL_PATH = "https://drive.google.com/file/d/1IsCAWfcMqP7Rao8qX5Jlw0n7O3JYVmCz/view?usp=sharing"


def load_model():
    "Load the pretrained and transfer learned model"
    
    global model
    model = torch.load(MODEL_PATH)
    model.eval()
    print('Model loaded. Check http://127.0.0.1:5000/')

def preprocess_image(image, target_size=(224,224)):
    "Preprocess a given image and transform it before feed to the model"
    
    if image.mode != 'RGB':
        image = image.convert("RGB")
    image = transforms.Resize(target_size)(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
    image = image[None]
    return torch.autograd.Variable(image, volatile=True)

def get_prediction(image_path):
    image = open(image_path, 'rb').read()
    img_dict = {'image':image}

    r = requests.post(REST_API_URL, files=img_dict).json()

    for (i, result) in enumerate(r['predictions'])
        print('This is {} with {:.4f} probability'.format(result['label'],
                                                          result['probability']))

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/predict", methods=['GET','POST'])
def predict():

    if flask.request.method == 'POST':
        if flask.request.files.get['file']:
            image = flask.request.files['file']
            image = Image.open(io.BytestIO(image))
            image = preprocess_image(image)

            prediction = F.softmax(model(image))
            results = torch.topk(prediction.data, k=2)

            data = {}
            data['predictions'] = list()
            labels = ["Hot Dog", "Not Hot Dog"]

            for prob, label in zip(results[0][0], results[1][0]):
                label_name = labels[label]
                r = {"label": label_name, "probability": float(prob)}
                data['predictions'].append(r)

    return flask.jasonify(data)

if __name__ == '__main__':
	print("Loading PyTorch Hotdog Classifier Model...")
	load_model()
	app.run()


