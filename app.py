from flask import Flask, request
from classify_nsfw import test_score
import json

app = Flask(__name__)
model_def = 'nsfw_model/deploy.prototxt'
pretrained_model = 'nsfw_model/resnet_50_1by2_nsfw.caffemodel'
img_key = 'img'


@app.route('/', methods=['POST'])
def hello_world():
    img = request.files[img_key]
    return json.dumps({'score': test_score(model_def, pretrained_model, img.read())})


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=8080,
        debug=True
    )
