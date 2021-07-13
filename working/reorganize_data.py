import numpy as np
from matplotlib import pyplot as plt
import os,sys,glob
import scipy.optimize as spo
import shutil

out_root = 'data_reorganized'
os.makedirs(out_root,exist_ok=True)

dsets = glob.glob('data/*')

in_types = ['drusDis','img','isosAng','isosLoc','isosVal']

for idx,dset in enumerate(dsets):
    base_dn = os.path.split(dset)[1]
    out_dn = os.path.join(out_root,'subject_%02d'%idx)
    os.makedirs(out_dn,exist_ok=True)

    # get the in and out directories
    in_dn = glob.glob(os.path.join(dset,'*_IN'))[0]
    out_dn = glob.glob(os.path.join(dset,'*_OUT'))[0]

    
    # get the pupil positions
    temp = glob.glob(os.path.join(in_dn,'*drusDis.npy'))
    temp = [os.path.split(t)[1] for t in temp]
    temp.sort()
    pupil_positions = []
    for t in temp:
        i1 = t.find('_PPos')+5
        i2 = t.find('_drusDis.npy')
        numstr = t[i1:i2]
        try:
            float(numstr)
        except:
            sys.exit('problem with pupil position string')
        pupil_positions.append(numstr)

    pupil_positions.sort(key=lambda x: float(x))

    for pupil_position in pupil_positions:
        for in_type in in_types:
            print(in_type)
            fn = glob.glob(os.path.join(in_dn,'*PPos%s_%s.npy'%(pupil_position,in_type)))[0]
            temp = np.load(fn)
            print(temp.shape)
            print(temp)
        print()
