import librosa
import json
import os
import math

DATA_PATH = "/Users/suyashramteke/PycharmProjects/Speech_Recognition_Design_to_Deployment/speech_data"
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
TRACK_DUR = 1
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DUR

def prepare_dataset(dataset_path, json_path, num_mfcc = 13, n_fft = 2048, hop_length = 512):
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": [],
        "files": []

    }

    num_mfcc_vectors_per_track = math.ceil(SAMPLES_PER_TRACK/hop_length)

    # Loop through the entire dataset. 'i' is the variable which is 0 when it is in speech data, 1 when it is down
    # and so on
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        if dirpath is not dataset_path:

            label = dirpath.split("/")[-1]
            data["mapping"].append(label)

            for f in filenames:
                file_path = os.path.join(dirpath, f)

                signal, s_r = librosa.load(file_path)
                # drop audio files with less than pre-decided number of samples
                if len(signal) >= SAMPLES_PER_TRACK:

                    # ensure consistency of the length of the signal
                    signal = signal[:SAMPLES_PER_TRACK]

                    # Hop length how big the segment would be in terms of frames
                    mfcc = librosa.feature.mfcc(signal[:], s_r, n_mfcc=num_mfcc, n_fft = n_fft, hop_length = hop_length)
                    mfcc = mfcc.T

                    # Ensure consistency of the dataset
                    if(len(mfcc) == num_mfcc_vectors_per_track):
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        data['files'].append(file_path)
                        print("{}: {}".format(file_path, i-1))

    with open (json_path, 'w') as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    prepare_dataset(DATA_PATH, JSON_PATH, num_mfcc=16, n_fft=2048, hop_length=512)






