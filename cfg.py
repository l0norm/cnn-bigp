#nfilt is the number of filters for mel-filterbanks its a choice
# nfeat is the number of extracted features from filterbanks
# all used in MFCCS
import os 
class Config:
    def __init__(self,mode='conv',nfilt=26,nfeat=13,nfft=512, rate=16000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfft = nfft
        self.nfeat = nfeat
        self.rate = rate
        self.step = int(rate/10)
        self.model_path = os.path.join('models', mode + '.keras')   #save weights and biases 
        self.p_path = os.path.join('pickles', mode + '.h5')         #pickle object to store mfcc features and not recalculate them 