import os 
from tqdm import tqdm #you can watch for loops as it is done
import pandas as pd
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt 
from scipy.io import wavfile
from python_speech_features import mfcc

import pickle # to save weights 
from keras.callbacks import ModelCheckpoint
from cfg import Config
from sklearn.model_selection import train_test_split

#to load weights from an object pickle from a file 
def check_data():
    if os.path.isfile(config.p_path):
        print('loading data for {} model '.format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else: 
        return None

def build_rand_feat():
    tmp = check_data()
    # returning previouse saved data 
    if tmp:
        return tmp.data[0], tmp.data[1]
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')# minimizing and maximizing 
    for _  in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.label==rand_class].index)
        rate, wav = wavfile.read('clean/'+file)
        print(wav.shape)
        label = df.at[file,'label']
        rand_index = np.random.randint(0,wav.shape[0]-config.step)# getting a random index from the sample data ... -config.step so if it was in last (goes back)
        sample = wav[rand_index:rand_index+config.step] #from the random index to the step your taking 
        X_sample = mfcc(sample, rate, # extracting mfcc features from the sample data
                        numcep=config.nfeat, nfilt = config.nfilt, nfft=config.nfft)
        
        X.append(X_sample)#for cnn(sample , collumn=features, row= time dimension) , RNN(samle , column = time, rows = features) cause it processes data over time 
        y.append(classes.index(label)) #all labels turned to index 
        config.min = _min = min(np.amin(X_sample), _min) # getting the min,max values from sample to normalize 
        config.max = _max = max(np.amax(X_sample), _max)
    X, y = np.array(X) ,np.array(y)
    X = (X - _min) / (_max - _min) # normalizing , between 0 and 1
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1) #conv = samle, heights, width, channels(1for gray scale, 3 for rgb) ,,, single channel spectogram
    elif config.mode == 'time':
        X =X.reshape(X.shape[0], X.shape[1], X.shape[2]) # time-series = sample, timesteps, features
    y = to_categorical(y, num_classes=10) #converts it one-hot encoding format 
    config.data = (X, y)

    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=-1)
    return X, y

def get_conv_model():
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation='relu', strides=(1,1), #16 filter and 3x3kernel 
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(32,(3,3), activation='relu', strides=(1,1),
                     padding='same'))
    model.add(Conv2D(64,(3,3), activation='relu', strides=(1,1),
                     padding='same'))
    model.add(Conv2D(128,(3,3), activation='relu', strides=(1,1),
                    padding='same'))
    
    model.add(MaxPool2D(2,2)) #good for reducing computation and not overfitting
    model.add(Dropout(0.5))   #disabling 50% of neurons for not overfitting 
    model.add(Flatten())       #after the cnn we need a fullly connected layer , so we converte it to 1d 
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',  
                  optimizer='adam', #way of optimizing 
                  metrics=['acc']) #traking accuracy 
    return model

def get_recurrent_model():
    # shape of data for RNN is (n, time, feature)
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(16, activation='relu')))
    model.add(TimeDistributed(Dense(8, activation='relu')))
    model.add(Flatten())
    model.add(Dense(10,activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['acc'])
    return model    






df = pd.read_csv('reciters.csv')
df.set_index('fname',inplace =True)

for f in df.index:
    rate, signal = wavfile.read('clean/'+f)
    df.at[f,'length'] = signal.shape[0]/rate  

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()#for everyclass its length mean

#sample data to feet our machine ,,,
n_samples = 2 * int(df['length'].sum()/0.1)#after getting the length we /0.1 to convert the number of segment of length 0.1s
prob_dist = class_dist / class_dist.sum() #probability for every class
choices = np.random.choice(class_dist.index, p= prob_dist) #random choice on probability


fig, ax = plt.subplots()
ax.set_title('Class Distribution', y = 1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
        shadow=False, startangle=90)
ax.axis('equal')


# ===============================Module==================================


config = Config(mode='conv')

if config.mode == 'conv':
    X, y = build_rand_feat()        #builds a random feature set from the data 
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_conv_model()


if config.mode == 'time':
    X, y = build_rand_feat()
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2])
    model = get_recurrent_model()

# to give each class weigts 
class_weight = compute_class_weight('balanced',
                                classes=np.unique(y_flat),
                                y=y_flat)


class_weight_dict = dict(zip(np.unique(y), class_weight))

checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc', verbose=1, mode='max',
                             save_best_only=True, save_weights_only=False, save_freq='epoch')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train, epochs=20, batch_size=32,
          shuffle=True,
          class_weight=class_weight_dict, validation_data=(X_test,y_test),
          callbacks=[checkpoint])

model.save(config.model_path)