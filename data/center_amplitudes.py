import glob,os
import numpy as np

flist = glob.glob(os.path.join('subject*','directionality_raw_data*'))

for f in flist:
    dat = np.load(f)
    dat[:,0] = dat[:,0]-np.mean(dat[:,0])
    outf = f.replace('.npy','_centered.npy')
    np.save(outf,dat)
