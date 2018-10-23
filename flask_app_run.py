from __future__ import division, print_function
import io
import os

from flask import Flask, request, render_template

import torch
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = None
MODEL_PATH = os.path.join(os.getcwd(),'models/model.pt')


def load_model():
    "Load the pretrained and transfer learned model"
    
    global model
    model = torch.load(MODEL_PATH)
    model.eval()
    # print('Model loaded. Check http://127.0.0.1:5000/')

def preprocess_image(image, target_size=(224,224)):
    "Preprocess a given image and transform it before feed to the model"
    
    if image.mode != 'RGB':
        image = image.convert("RGB")
    image = transforms.Resize(target_size)(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
    image = image[None]
    return torch.autograd.Variable(image, volatile=True)

@app.route('/', methods=['GET'])
def index():
    return render_template('templates/index.html')


@app.route("/predict", methods=['GET','POST'])
def predict():

    if request.method == 'POST':
        if request.files['file']:
            f = flask.request.files['file']

            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'images', secure_filename(f.filename))
            f.save(file_path)

            image = Image.open(file_path)
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

            for (i, result) in enumerate(data['predictions']):
                print('This is {} with {:.4f} probability.'.format(result['label'],
                                                          result['probability']))
    return None

if __name__ == '__main__':
    print("Loading PyTorch Hotdog Classifier Model...")
    app.run(port=5002, debug=True)
    load_model()
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()


