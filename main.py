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
    for c in test_class:
    		for i in m.features[c]:
    				train_set.append(i)
    				
    # construct autoencoder
    train_data = T.dmatrix('train_data')
    x = T.dmatrix('x')  # the data is presented as rasterized images
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 10))
    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=39,
        n_hidden=100
    )
    cost, updates = da.get_cost_updates(
        corruption_level=0.,
        learning_rate=0.4
    )
    train_da = theano.function(
    		[train_data],
        cost,
        updates=updates,
        givens={
            x: train_data
        }
    )
    # train autoencoder
    training_epochs = 100
    c1 = []
    for epoch in xrange(training_epochs):
    		c1.append(train_da(numpy.asarray(train_set)))
    		print 'Training epoch %d, cost ' % epoch, c1[len(c1)-1]
    
    for c in test_class:
    		m.features[c] = da.get_hidden_values(m.features[c]).eval()
    
    m.train()
    m.dump('model/model_da.out')
    with open('model/da.out', 'w') as f:
    		pickle.dump(da, f, -1)

def test( ):
    m = ModelInterface.load('model/model_da.out')
    with open('model/da.out', 'r') as f:
    		da = pickle.load(f)
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
    				#if(m.predict(fs, signal[:80000]) == c):
    				if(predict(m, fs, signal[:80000], da) == c):
    					count = count + 1
    print 'accuracy is:', (100.0*count)/(len(test_class)*len(file_name)), '%'

def predict(m, fs, signal, da):
        try:
            feat = mix_feature((fs, signal))
            feat = da.get_hidden_values(feat).eval()
        except Exception as e:
            return None
        return m.gmmset.predict_one(feat)

if __name__ == '__main__':
    #train()
    feature_re_extract()
    test()
