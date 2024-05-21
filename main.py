import numpy as np
from PIL import Image
from keras.models import load_model # type: ignore
from keras.applications.vgg19 import preprocess_input # type: ignore
from flask import Flask, request, render_template, jsonify, send_file
import requests

file_name = "/"

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/result', methods=['POST'])
def upload_file():
    global file_name
    if request.method == 'POST':
        file = request.files['file']
        file.save("image/"+file.filename) # save image file
        file_name += file.filename # filename up to global variable
        
        # import API from IBM Cloud
        API_KEY = "31jWY7WlgffXvxsph_6ia0g12SqWu9LdhSGiXlBwnEAi"
        token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
        mltoken = token_response.json()["access_token"]
        
        # process image clasifications
        output_class = ["battery", "glass", "metal", "organic", "paper", "plastic"]
        def preprocessing_input(img_path):
            img = Image.open(img_path)
            img = img.resize((224, 224))
            img = np.array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            return img
        
        # prediction image classification
        def predict_user(img_path):
            try:
                img = preprocessing_input(img_path)
                # load model from ibm cloud
                payload_scoring = {"input_data": [{"values": img.tolist()}]}
                header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
                scoring_url = ''
                response_scoring = requests.post(scoring_url, json=payload_scoring, headers=header)
                
                # response result from IBM
                result = response_scoring.json()['predictions'][0]['values'][0]
                predicted_class_idx = np.argmax(result)
                predicted_class = output_class[predicted_class_idx]
                predicted_probability = result[predicted_class_idx]
                callback = {
                    "accuracy": f"{predicted_probability:.2%}",
                    "category": predicted_class,
                    "name": file.filename
                }
                print(callback,"\n")
                return callback
            except:
                callback_error = {
                    "accuracy": "-%",
                    "category": "Not found",
                    "name": file.filename
                    }
                print(callback_error,"\n")
                return callback_error   
        try:
            return jsonify(predict_user("image/"+file.filename))#, os.remove("img/"+file.filename)
        except:
            return "<h1>Error System</h1>"#, os.remove("img/"+file.filename)

@app.route('/<file_name>')
def image(file_name):
    path_image = f"./image/{file_name}"
    return send_file(path_image)

if __name__ == '__main__':
    app.run(port=5000)




