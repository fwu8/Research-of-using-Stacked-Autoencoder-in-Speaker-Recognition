#  from https://github.com/ppwwyyxx/speaker-recognition/
import time
import os
import sys
import numpy as np
import cPickle as pickle
import traceback as tb
import pdb
from os import path
from collections import defaultdict
from scipy.io import wavfile

from feature import mix_feature
from filters.VAD import VAD
from skgmm import GMMSet, GMM

CHECK_ACTIVE_INTERVAL = 1

class ModelInterface(object):

    UBM_MODEL_FILE = 'model/ubm.mixture-32.utt-300.model'

    def __init__(self):
        self.features = defaultdict(list)
        self.gmmset = GMMSet()
        self.vad = VAD()

    def init_noise(self, fs, signal):
        self.vad.init_noise(fs, signal)

    def filter(self, fs, signal):
        ret, intervals = self.vad.filter(fs, signal)
        orig_len = len(signal)

        if len(ret) > orig_len / 3:
            # signal is filtered by VAD
            return ret
        return np.array([])

    def enroll(self, name, fs, signal):
        feat = mix_feature((fs, signal))
        #pdb.set_trace()
        self.features[name].extend(feat)

    def _get_gmm_set(self):
        if os.path.isfile(self.UBM_MODEL_FILE):
            try:
                from gmmset import GMMSetPyGMM
                if GMMSet is GMMSetPyGMM:
                    return GMMSet(ubm=GMM.load(self.UBM_MODEL_FILE))
            except Exception as e:
                print "Warning: failed to import gmmset. You may forget to compile gmm:"
                print e
                print "Try running `make -C src/gmm` to compile gmm module."
                print "But gmm from sklearn will work as well! Using it now!"
            return GMMSet()
        return GMMSet()

    def train(self):
        self.gmmset = self._get_gmm_set()
        start = time.time()
        print "Start training..."
        for name, feats in self.features.iteritems():
            self.gmmset.fit_new(feats, name)
        print time.time() - start, " seconds"

    def predict(self, fs, signal):
        try:
            feat = mix_feature((fs, signal))
        except Exception as e:
            print tb.format_exc()
            return None
        return self.gmmset.predict_one(feat)

    def dump(self, fname): # save model
        self.gmmset.before_pickle()
        with open(fname, 'w') as f:
            pickle.dump(self, f, -1)
        self.gmmset.after_pickle()

    @staticmethod
    def load(fname): # get model
        with open(fname, 'r') as f:
            R = pickle.load(f)
            R.gmmset.after_pickle()
            return R

if __name__ == "__main__":
    m = ModelInterface()
    fs, signal = wavfile.read("data/5_1.wav")
    m.enroll('1', fs, signal[:80000])
    fs, signal = wavfile.read("data/5_2.wav")
    m.enroll('2', fs, signal[:80000])
    m.train()
    
    fs, signal = wavfile.read("data/5_1.wav")
    print m.predict(fs, signal[:80000])
