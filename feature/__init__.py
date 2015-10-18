


#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: __init__.py
# Date: Sat Nov 29 21:42:15 2014 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

#  from https://github.com/ppwwyyxx/speaker-recognition/
import pdb
import sys
import numpy
try:
    import BOB as MFCC
except:
    print >> sys.stderr, "Warning: failed to import Bob, use a slower version of MFCC instead."
    import MFCC
import LPC
import numpy as np

def get_extractor(extract_func, **kwargs):
    def f(tup):
        return extract_func(*tup, **kwargs)
    return f
    
def differentiate(feature):
		res = []
		for i in xrange(len(feature)):
			r = []
			for k in xrange(len(feature[0])-1):
				r.append(feature[i,k+1] - feature[i, k])
			res.append(r)
		res = numpy.asarray(res)
		return res
			
def mix_feature(tup):
    mfcc = MFCC.extract(tup)
    lpc = LPC.extract(tup)
    mfcc_1dif_coef = differentiate(mfcc)
    mfcc_2dif_coef = differentiate(mfcc_1dif_coef)
    if len(mfcc) == 0:
        print >> sys.stderr, "ERROR.. failed to extract mfcc feature:", len(tup[1])
    #pdb.set_trace()
    #return np.concatenate((mfcc, lpc), axis=1) # 28 dimension
    # 39 dimension: mfcc 0-12 coefficient, and 13 first-order differential coefficient
    # 13 second-order differential coefficient
    return np.concatenate((mfcc[:,0:13], mfcc_1dif_coef[:,0:13], mfcc_2dif_coef), axis=1) 
    
