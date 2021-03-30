from flask import Flask, request, jsonify
import random
import os
from predict import Keyword_Spotting_Service

# Instantiating a flask application
app = Flask(__name__)

# Request
@app.route("/predict", methods = ['POST'])

def predict():

    # Get file from POST request and save it
    audio_file = request.files["file"]
    file_name = str(random.randint(0, 100000))
    audio_file.save(file_name)

    # instantiate keyword spotting service singleton and get prediction
    kss = Keyword_Spotting_Service()
    predicted_keyword = kss.predict(file_name)

    # Remove the audio file
    os.remove(file_name)

    # Send back the result in a json format
    result = {"keyword": predicted_keyword}
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=False)



