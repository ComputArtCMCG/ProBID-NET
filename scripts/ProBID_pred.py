import numpy as np
import os
import sys
import h5py
import pandas as pd
from tensorflow.python import keras
from tensorflow.python.keras import models
from tensorflow.python.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import argparse
from argparse import RawTextHelpFormatter
parser = argparse.ArgumentParser()

three2int = {'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4, 'GLY': 5,
              'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9, 'MET': 10, 'ASN': 11,
              'PRO': 12, 'GLN': 13, 'ARG': 14, 'SER': 15, 'THR': 16, 'VAL': 17,
              'TRP': 18, 'TYR': 19}
int2one= {0:'A',1: 'C',2: 'D',3: 'E', 4:'F',  5:'G',
            6: 'H',7: 'I', 8:'K', 9:'L',10: 'M', 11:'N',
            12:'P',13: 'Q', 14:'R', 15:'S', 16:'T', 17:'V',
            18:'W', 19:'Y'}

def aaname(resname):
    aaname={'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G',
            'HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N',
            'PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V',
            'TRP':'W','TYR':'Y'}
    try:
        aa = aaname[resname]
    except:
        raise Exception("unknow resname %s" % resname)
        aa = "X"
    return aa


def read_hdf5(input):
    x_train=[]
    y_train=[]
    z_train=[]
    with h5py.File(input, 'r') as h5f:
        for key in h5f.keys():
            restype, resid, chain = key.split('_')
            m = h5f[key]
            n = three2int[restype]
            x_train.append(m)
            y_train.append(n)
            z_train.append(restype+' '+ resid+' '+ chain)
        X=np.array(x_train,dtype=np.float32)
        Y = np.array(y_train, dtype=np.int64)
        Y = to_categorical(Y)
        Z=z_train
    return X, Y ,Z

def prob_dic(res):
    dic={}
    for i in range(20):
        dic[int2one[i]]=round(res[int(i)],4)
    return dic

def make_prediction(model, fhdf5 ,outpath):
    #reslist_test,resid-get_reslist(fhdf5)
    x,y,z=read_hdf5(fhdf5)
    print("get", len(y), "interface residues from", fhdf5)
    #model=load_model(model)
    for m in range(len(model)):
        pred=model[m].predict(x)
        pred=pred.tolist()
    #save probility in pro_file
        for i in range(len(pred)):
            res_position=z[i]
            prob=[]
            aa=[]
            dic=prob_dic(pred[i])
            mi = dict(zip(dic.values(), dic.keys()))
    #print(mi)
            for value in dic.values():
                prob.append(value)
            prob.sort(reverse=True)
            for i in prob:
                aa.append(mi[i])
    #print(aa)
    #print(prob)
            zipped= zip(aa,prob)
            ziplist=list(zipped)
    #print(ziplist)
            with open(outpath+'_'+str(m),'a') as t:
                t.write(res_position+' '+'   '.join('%s %s' % x for x in ziplist) +'\n')


if __name__=='__main__':
    des='########################### Perform prediction ####################################'
    parser = argparse.ArgumentParser(description=des, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-predict', action="store", dest="predict_pdbs", required=True,help="predict mode,input list of pdbid")
    parser.add_argument('-model', action="append", dest="model",required=True,help="0 means picking domain-added model,1-5 means fold model")
    parser.add_argument('-datapath', action="store", dest="datapath", required=True, \
                        help="path to the input complex pdb h5 model files")
    parser.add_argument('-outpath', action="store", dest="outpath", required=True, \
                        help="path to output the prediction file")
    parser.add_argument('-batchsize', action="store", dest="batchsize", required=True, default=8, \
                        help="batchsize", type=int)

    print("Command line:", " ".join(sys.argv))
    inputarg = parser.parse_args()

    model=[load_model(m,compile=False) for m in inputarg.model]
    #print(model)
    for line in open(inputarg.predict_pdbs):
        line=line.split()
        if len(line) == 0:
            continue
        if line[0][0] == "#":
            continue
        pdbid = line[0]
        chain=line[1]

        fhdf5  = os.path.join(inputarg.datapath,pdbid+'.'+chain+'.hdf5')
        outpath = os.path.join(inputarg.outpath, pdbid+'_'+chain + ".pred")
        make_prediction(model, fhdf5, outpath)
        print(pdbid+' has been been successfully predicted and saved in '+outpath)
