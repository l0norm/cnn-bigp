import pickle
import os
import numpy as np
from tqdm import tqdm 
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score

# def build_predications(audio_dir):
#     y_true = []
#     y_pred = []
#     fn_prob = {}

#     print("extracting features ")
#     for fn in tqdm(os.listdir(audio_dir)):
#         rate, wav = wavfile.read(os.path.join(audio_dir, fn))
#         label = fn2class[fn]
#         c = classes.index(label)   #getting index of specific class to compare with output 
#         y_prob = []

#         for i in range(0,wav.shape[0]-config.step, config.step):
#             sample = wav[i:i+config.step]
#             X = mfcc(sample, rate, 
#                         numcep=config.nfeat, nfilt = config.nfilt, nfft=config.nfft)
#             X = (X - config.min) / (config.max - config.min)
            
#             if config.mode == 'conv':
#                 X = X.reshape(1,X.shape[0],X.shape[1],1)
#             elif config.mode == 'time':
#                 X = np.expand_dins(X,axis=0)

#             y_hat = model.predict(X)

#             y_prob.append(y_hat)
#             y_pred.append(np.argmax(y_hat))

#             y_true.append(c)
        
#         fn_prob[fn] = np.mean(y_prob, axis=0).flatten()

#     return y_true, y_pred, fn_prob

def build_prediction(audio_file):
    y_true = []
    y_pred = []
    fn_prob = {}

    rate, wav = wavfile.read(audio_file)
    label = 'Aadel_Alkalbani_(MP3_Quran)'
    c = classes.index(label)

    y_prob = []     #probability for the audio file for each sample
    print('extracting features')
    for i in range(0,wav.shape[0]-config.step, config.step):
        sample = wav[i:i+config.step]
        X = mfcc(sample, rate, 
                    numcep=config.nfeat, nfilt = config.nfilt, nfft=config.nfft)
        X = (X - config.min) / (config.max - config.min)
        
        if config.mode == 'conv':
            X = X.reshape(1,X.shape[0],X.shape[1],1)
        elif config.mode == 'time':
            X = np.expand_dins(X,axis=0)
        
        print(X.shape)
        
        y_hat = model.predict(X)

        y_prob.append(y_hat)
        y_pred.append(np.argmax(y_hat))

        y_true.append(c)

    fn_prob[audio_file] = np.mean(y_prob, axis=0).flatten()
    
    return y_true, y_pred, fn_prob






df = pd.read_csv("reciters.csv")
classes = list(np.unique(df.label))
fn2class = dict(zip(df.fname, df.label))
p_path = os.path.join('pickles', 'conv.h5')

with open(p_path, 'rb') as handler:
    config = pickle.load(handler)


model = load_model(config.model_path)

audio_file = 'output_test.wav'
y_true, y_pred, fn_prob = build_prediction(audio_file)



acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)

y_prob = fn_prob[audio_file]
predicted_class = classes[np.argmax(y_prob)]
df.loc[df['fname'] == os.path.basename(audio_file), classes] = y_prob
df.loc[df['fname'] == os.path.basename(audio_file), 'y_pred'] = predicted_class


print(f"Predicted Class for {audio_file}: {predicted_class}")
print(f"Prediction Probabilities: {y_prob}")
print(f"Accuracy Score: {acc_score}")


# y_probs = []
# for i,row in df.iterrows():
#     y_prob = fn_prob[row.fname]
#     y_probs.append(y_prob)
#     for c,p in zip(classes, y_prob):
#         df.at[i,c] = p

# y_pred = [classes[np.argmax(y)] for y in y_probs]
# df['y_pred'] = y_pred

# df.to_csv('predictions.csv')