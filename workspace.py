import numpy as np
from matplotlib import pyplot as plt
import os,sys,glob
import scipy.optimize as spo
import shutil

plt.style.use('seaborn-deep')

data_root = 'data_reorganized'

dsets = glob.glob(os.path.join(data_root,'*'))


def rolling_average(theta,amp,window_width=4.0,step_size=2.0,diagnostics=False):
    
    # find the start and end thetas
    t_start = np.min(theta)
    t_end = np.max(theta)

    window_centers = []
    amplitude_mean = []
    amplitude_std = []

    for t in np.arange(t_start,t_end+step_size,step_size):
        window_centers.append(t+window_width/2.0)

        # find he indices in the theta array where the angle falls in our window
        idx = np.where(np.logical_and(theta>=t,theta<t+window_width))[0]

        amplitude_mean.append(np.mean(amp[idx]))
        amplitude_std.append(np.std(amp[idx]))

    if diagnostics:
        pl.figure()
                            
        # plot the raw data with low alpha
        plt.plot(theta,amp,'k.',alpha=0.1,markersize=1,label='raw data')

        # plot the average with standard deviation bars
        plt.errorbar(window_centers,amplitude_mean,amplitude_std,label='rolling average')

        plt.legend()

    return np.array(window_centers),np.array(amplitude_mean),np.array(amplitude_std)
    
for dset in dsets:
    subject_id = os.path.split(dset)[1]
    if subject_id[0]=='_':
        continue
    
    raw_data_drusen = np.load(os.path.join(dset,'directionality_raw_data_drusen.npy'))
    raw_data_nondrusen = np.load(os.path.join(dset,'directionality_raw_data_nondrusen.npy'))

    theta_rd_d = raw_data_drusen[:,0]
    theta_rd_nd = raw_data_nondrusen[:,0]
    amp_rd_d = raw_data_drusen[:,1]
    amp_rd_nd = raw_data_nondrusen[:,1]

    #rolling_average(theta_rd_nd,amp_rd_nd)
    
    #sys.exit()
    
    rolling_average_drusen = np.load(os.path.join(dset,'directionality_rolling_average_drusen.npy'))
    fitted_curve_drusen = np.load(os.path.join(dset,'directionality_fitted_curve_drusen.npy'))
    rolling_average_nondrusen = np.load(os.path.join(dset,'directionality_rolling_average_nondrusen.npy'))
    fitted_curve_nondrusen = np.load(os.path.join(dset,'directionality_fitted_curve_nondrusen.npy'))

    plt.figure()
    plt.plot(raw_data_nondrusen[:,0],raw_data_nondrusen[:,1],'k.',alpha=0.1,markersize=1,label='raw data')
    plt.plot(rolling_average_nondrusen[:,0],rolling_average_nondrusen[:,1],'b-',label='rolling average')

    wc,mamp,samp = rolling_average(theta_rd_nd,amp_rd_nd)
    plt.errorbar(wc,mamp,samp)
    
    #plt.plot(fitted_curve_drusen[:,0],fitted_curve_drusen[:,1],'k--')
    #plt.plot(rolling_average_nondrusen[:,0],rolling_average_nondrusen[:,1],'b--')
    #plt.plot(fitted_curve_nondrusen[:,0],fitted_curve_nondrusen[:,1],'k--')
    plt.title(subject_id)
    
plt.show()

print(dsets)
sys.exit()

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
