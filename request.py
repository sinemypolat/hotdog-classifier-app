import requests
import argparse

REST_API_URL = 'http://127.0.0.1:5000/predict'

def get_prediction(image_path):
    image = open(image_path, 'rb').read()
    img_dict = {'image':image}

    r = requests.post(REST_API_URL, files=img_dict).json()

    for (i, result) in enumerate(r['predictions'])
        print('This is {} with {:.4f} probability'.format(result['label'],
                                                          result['probability']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hotdog-classifier')
    parser.add_argument('--file', type=str, help='upload a test image path')

    args = parser.parse.args()
    get_prediction(args.file)