from flask import Flask, request
from flask_cors import CORS
import model
import json

app = Flask(__name__)
cors = CORS(app)

@app.route('/upload', methods=['POST'])
def receive_image():
    file = request.files.get('file')

    return json.dumps(model.guess(file.stream))