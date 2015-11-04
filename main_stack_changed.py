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
import cPickle
from theano.tensor.shared_randomstreams import RandomStreams
from autoencoder.dA import dA
from interface import ModelInterface
from utils import read_wav
from filters.silence import remove_silence
from feature import mix_feature
from mSdA import mSdA
import pickle

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
    				m.enroll(c, fs, signal[:80000])
    m.train()
    m.dump('model/model.out')

def feature_re_extract():
    #pdb.set_trace()
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
    '''
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
    '''
    # construct stacked autoencoder
    sda = mSdA(
    		layers = [39, 100]
    )
    sda.train(train_set)
    
    for c in test_class:
    		'''
    		for i in xrange(len(m.features[c])):
    				for j in xrange(len(m.features[c][i])):
    						m.features[c][i][j] = 2*(m.features[c][i][j]-lower_bound[j])/(up_bound[j]-lower_bound[j])-1
    		'''
    		m.features[c] = sda.get_hidden_values(m.features[c])
    
    m.train()
    m.dump('model/model_sda.out')
    with open('model/sda.out', 'w') as f:
    		pickle.dump(sda, f, -1)
    return up_bound, lower_bound

def test( up_bound, lower_bound ):
    m = ModelInterface.load('model/model_sda.out')
    with open('model/sda.out', 'r') as f:
    		sda = pickle.load(f)
    count = 0;
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
    				if(predict(m, fs, signal[:80000], sda, up_bound, lower_bound) == c):
    						count = count + 1
    print 'accuracy is:', (100.0*count)/(len(test_class)*len(file_name)), '%'

def predict(m, fs, signal, sda, up_bound, lower_bound):
        try:
            feat = mix_feature((fs, signal))
            '''
            # put all values into -1~1
            for i in xrange(len(feat)):
    						for j in xrange(len(feat[0])):
    								feat[i][j] = 2*((feat[i][j]-lower_bound[j]) / (up_bound[j]-lower_bound[j]))-1
    				'''
            feat = sda.get_hidden_values(feat)
        except Exception as e:
            return None
        return m.gmmset.predict_one(feat)

if __name__ == '__main__':
    #train()
    up_bound, lower_bound = feature_re_extract()
    test( up_bound, lower_bound )
