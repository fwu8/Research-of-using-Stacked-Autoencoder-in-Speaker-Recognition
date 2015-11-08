import pdb
import argparse
import sys
import glob
import os
import itertools
import scipy.io.wavfile as wavfile
import theano
import theano.tensor as T
import numpy
from theano.tensor.shared_randomstreams import RandomStreams
from autoencoder.dA import dA
from interface import ModelInterface
from utils import read_wav
from filters.silence import remove_silence
from feature import mix_feature
from mSdA import mSdA

def train( ):
    m = ModelInterface()
    train_dir = 'data/train/'
    train_class = ['FAML_S', 'FDHH_S', 'FEAB_S', 'FHRO_S', 
    		'FJAZ_S', 'FMEL_S', 'FMEV_S', 'FSLJ_S', 'FTEJ_S', 
    		'FUAN_S', 'MASM_S', 'MCBR_S', 'MFKC_S', 'MKBP_S', 
    		'MLKH_S', 'MMLP_S', 'MMNA_S', 'MNHP_S', 'MOEW_S',
    		'MPRA_S', 'MREM_S', 'MTLS_S']
    file_name = ['a.wav', 'b.wav', 'c.wav', 'd.wav', 'e.wav', 'f.wav', 'g.wav']
    for c in train_class:
    		for n in file_name:
    				fs, signal = wavfile.read(train_dir + c + n)
    				m.enroll(c, fs, signal)
    m.train()
    m.dump('model/model.out')

def feature_re_extract():
    test_class = ['FAML_S', 'FDHH_S', 'FEAB_S', 'FHRO_S', 
    		'FJAZ_S', 'FMEL_S', 'FMEV_S', 'FSLJ_S', 'FTEJ_S', 
    		'FUAN_S', 'MASM_S', 'MCBR_S', 'MFKC_S', 'MKBP_S', 
    		'MLKH_S', 'MMLP_S', 'MMNA_S', 'MNHP_S', 'MOEW_S',
    		'MPRA_S', 'MREM_S', 'MTLS_S']
    m = ModelInterface.load('model/model.out')
    
    # construct train set
    train_set = []
    up_bound = []
    lower_bound = []
    for c in test_class:
    		for i in m.features[c]:
    				train_set.append(i)
    
    # put all values into -1~1
    up_bound = []
    lower_bound = []
    for j in xrange(len(train_set[0])):
    		up_bound.append(train_set[0][j])
    		lower_bound.append(train_set[0][j])
    for i in xrange(len(train_set)):
    		for j in xrange(len(train_set[0])):
    				up_bound[j] = max(up_bound[j], train_set[i][j])
    				lower_bound[j] = min(lower_bound[j], train_set[i][j])
    for i in xrange(len(train_set)):
    		for j in xrange(len(train_set[0])):
    				train_set[i][j] = 2*((train_set[i][j]-lower_bound[j]) / (up_bound[j]-lower_bound[j]))-1
    
    # construct stacked autoencoder
    sda = mSdA(
    		layers = [39, 100]
    )
    sda.setMinMax(up_bound, lower_bound)
    sda.train(train_set, 500) # use 500 as the batch size
    for c in test_class:
    		m.features[c] = sda.get_hidden_values(m.features[c])
    m.train()
    m.dump('model/model_sda.out')
    sda.dump('model/sda.out')

def test():
    m = ModelInterface.load('model/model_sda.out')
    sda = mSdA.load('model/sda.out')
    count = 0
    allsum = 0
    test_dir = 'data/test/'
    test_class = ['FAML_S', 'FDHH_S', 'FEAB_S', 'FHRO_S', 
    		'FJAZ_S', 'FMEL_S', 'FMEV_S', 'FSLJ_S', 'FTEJ_S', 
    		'FUAN_S', 'MASM_S', 'MCBR_S', 'MFKC_S', 'MKBP_S', 
    		'MLKH_S', 'MMLP_S', 'MMNA_S', 'MNHP_S', 'MOEW_S',
    		'MPRA_S', 'MREM_S', 'MTLS_S']
    file_name = ['1.wav', '2.wav']
    for c in test_class:
    		for n in file_name:
    				fs, signal = wavfile.read(test_dir + c + n)
    				signal_size = 40000
    				for indx in xrange(len(signal)/signal_size):
    						allsum = allsum + 1
    						if(predict(m, fs, signal[indx*signal_size:(indx+1)*signal_size], sda) == c):
    								count = count + 1
    print 'accuracy is:', (100.0*count)/(allsum), '%'

def predict(m, fs, signal, sda):
        try:
            feat = mix_feature((fs, signal))
            up_bound = sda.up_bound
            lower_bound = sda.lower_bound
            # put all values into -1~1
            for i in xrange(len(feat)):
    						for j in xrange(len(feat[0])):
    								feat[i][j] = 2*((feat[i][j]-lower_bound[j]) / (up_bound[j]-lower_bound[j]))-1
            feat = sda.get_hidden_values(feat)
        except Exception as e:
            return None
        return m.gmmset.predict_one(feat)

if __name__ == '__main__':
    #train()
    #feature_re_extract()
    test()
