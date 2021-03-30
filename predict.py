import librosa
import tensorflow as tf
import numpy as np

SAVED_MODEL_PATH = "model.h5"
SAMPLES_TO_CONSIDER = 22050

""" Singleton Class : We will be having only instance running in the class. 
    This saves a lot of computation and makes the process run faster  """
class _Keyword_Spotting_Service:
    model = None
    mapping = [
        "right",
        "go",
        "no",
        "left",
        "stop",
        "up",
        "down",
        "yes",
        "on",
        "off"
    ]

    _instance = None

    def predict(self, file_path):
        # Extract MFCC
        mfcc = self.preprocess(file_path)

        # Convert 2D array to 3D array [#num_samples, #segments, #coefficients]
        mfcc = mfcc[np.newaxis, ...]

        # Make predictions
        predictions = self.model.predict(mfcc)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self.mapping[predicted_index]

        return predicted_keyword

    def preprocess(self, file_path, num_mfcc=16, n_fft=2048, hop_length=512):
        # Load audio file
        signal, sample_rate = librosa.load(file_path)

        # Ensure length of track
        if(len(signal)) >= SAMPLES_TO_CONSIDER:
            signal = signal[:SAMPLES_TO_CONSIDER]

            mfcc = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc = mfcc.T

        return mfcc

def Keyword_Spotting_Service():

        # ensure an instance is created only the first time the factory function is called
        if _Keyword_Spotting_Service._instance is None:
            _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
            _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
        return _Keyword_Spotting_Service._instance

if __name__ == "__main__":
    kss = Keyword_Spotting_Service()
    kss1 = Keyword_Spotting_Service()

    # check that different instances of the keyword spotting service point back to the same object (singleton)
    assert kss is kss1

    # make a prediction
    keyword = kss.predict("/Users/suyashramteke/PycharmProjects/Speech_Recognition_Design_to_Deployment/speech_data/left/0ac15fe9_nohash_0.wav")
    print("Predicted word : ", keyword)



