import numpy as np
from matplotlib import pyplot as plt
import os,sys,glob
import scipy.optimize as spo
import shutil

out_root = 'data_reorganized'
os.makedirs(out_root,exist_ok=True)

dsets = glob.glob('data/*')

in_types_1 = ['drusDis','img','isosAng','isosLoc','isosVal']
out_types_1 = ['distance_to_drusen_margin','average_bscan','isos_angle','isos_axial_location','isos_amplitude']

in_types_2 = ['FitPts_Drusen', 'FitPts_NonDrusen', 'ParamFit_Drusen','ParamFit_NonDrusen','Pts_Drusen','Pts_NonDrusen','RollAvg_Drusen','RollAvg_NonDrusen']
out_types_2 = ['fitted_curve_drusen','fitted_curve_nondrusen','fitting_parameters_drusen','fitting_parameters_nondrusen','raw_data_drusen','raw_data_nondrusen','rolling_average_drusen','rolling_average_nondrusen']

for idx,dset in enumerate(dsets):
    base_dn = os.path.split(dset)[1]
    out_dn = os.path.join(out_root,'subject_%02d'%(idx+1))
    os.makedirs(out_dn,exist_ok=True)

    # get the in and out directories
    # confusing, but call both of these in_*, because we're copying data from them
    in_dn_1 = glob.glob(os.path.join(dset,'*_IN'))[0]
    in_dn_2 = glob.glob(os.path.join(dset,'*_OUT'))[0]

    
    # get the pupil positions
    temp = glob.glob(os.path.join(in_dn_1,'*drusDis.npy'))
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
        for in_type,out_type in zip(in_types_1,out_types_1):
            in_fn = glob.glob(os.path.join(in_dn_1,'*PPos%s_%s.npy'%(pupil_position,in_type)))[0]
            temp = np.load(in_fn)
            ppf = float(pupil_position)
            if ppf<0:
                pp_string = 'm%0.1f'%abs(ppf)
            else:
                pp_string = 'p%0.1f'%abs(ppf)
                
            out_fn = os.path.join(out_dn,'pupil_position_%s_%s.npy'%(pp_string,out_type))
            shutil.copyfile(in_fn,out_fn)
            print(in_fn,'->',out_fn)
            #print()
        #print()

    for in_type,out_type in zip(in_types_2,out_types_2):
        in_fn = glob.glob(os.path.join(in_dn_2,'*_%s.npy'%in_type))[0]
        out_fn = os.path.join(out_dn,'directionality_%s.npy'%out_type)
        shutil.copyfile(in_fn,out_fn)
        print(in_fn,'->',out_fn)
