import numpy as np
from matplotlib import pyplot as plt
import os,sys,glob
import scipy.optimize as spo
from scipy import stats

exclude2 = True
savefig = True

sublist = ['2018.04.17','2019.07.08','2019.07.24','2019.07.31']
drusen1, drusen2, drusen3, drusen4 = [np.load(os.path.join(os.path.join(os.path.join('data','%s_processed'%sub),'%s_OUT'%sub),'%s_RollAvg_Drusen.npy'%sub)) for sub in sublist]
normal1, normal2, normal3, normal4 = [np.load(os.path.join(os.path.join(os.path.join('data','%s_processed'%sub),'%s_OUT'%sub),'%s_RollAvg_NonDrusen.npy'%sub)) for sub in sublist]

def func(theta,a,b,rho,theta0):
    return b+a*10**(-1*rho*((0.0167*np.tan(np.radians(theta)))-(0.0167*np.tan(np.radians(theta0))))**2)

def gaussian_fit(mat):
    x = mat[:,0]
    y = mat[:,1]

    A_init = y.max()*0.8
    B_init = y.min()
    Rho_init = 1000
    Theta0_init = x[np.argmax(y)]
    
    A_bound = (y.max()*0.2,y.max()*2)
    B_bound = (y.min()*0.2,y.min()*1.2)
    Rho_bound = (0,np.inf)
    Theta0_bound_abs = max(abs(x.min()),abs(x.max()))
    Theta0_bound = (-Theta0_bound_abs,Theta0_bound_abs)

    init_guesses = [A_init,B_init,Rho_init,Theta0_init]
    init_bounds = [(A_bound[0],B_bound[0],Rho_bound[0],Theta0_bound[0]),
                   (A_bound[1],B_bound[1],Rho_bound[1],Theta0_bound[1])]


    popt,pcov = spo.curve_fit(func,x,y,init_guesses,bounds=init_bounds)

    return popt,pcov

def plot(mat,c='b',label=None):
    x = mat[:,0]
    y = mat[:,1]
    popt,pcov = gaussian_fit(mat)
    fit = func(x,*popt)
    
    A,B,rho,theta0 = popt
    
    print('%s\n\tA: %s\n\tB: %s\n\tRho: %s\n\tTheta0: %s\n' % (label,round(A,3),round(B,3),round(rho,3),round(theta0,3)))    

    return A,B,rho,theta0

print('\nEquation:     I(Theta) =       -Rho((0.0167*tan(Theta)-(0.0167*tan(Theta0))')
print('                         B+A*10\n')

fitvalues_normal = []
fitvalues_drusen = []

fitvalues_normal.append(plot(normal1,'r','normal1'))
fitvalues_drusen.append(plot(drusen1,'b','drusen1'))

#fitvalues_normal.append(plot(normal2,'r','normal2'))
#fitvalues_drusen.append(plot(drusen2,'b','drusen2'))

fitvalues_normal.append(plot(normal3,'r','normal3'))
fitvalues_drusen.append(plot(drusen3,'b','drusen3'))

fitvalues_normal.append(plot(normal4,'r','normal4'))
fitvalues_drusen.append(plot(drusen4,'b','drusen4'))

#================================================================================
#================================================================================

A_normal,B_normal,rho_normal,theta0_normal = list(map(list, list(zip(*fitvalues_normal))))
A_drusen,B_drusen,rho_drusen,theta0_drusen = list(map(list, list(zip(*fitvalues_drusen))))

names = ['N1','N3','N4','D1','D3','D']


print(A_normal)
print(B_normal)

print('ind A:/t', stats.ttest_ind(A_normal,A_drusen))
print('ind B:/t', stats.ttest_ind(B_normal,B_drusen))
print('ind Rho:/t', stats.ttest_ind(rho_normal,rho_drusen))

print('paired A:/t', stats.ttest_rel(A_normal,A_drusen))
print('paired B:/t', stats.ttest_rel(B_normal,B_drusen))
print('paired Rho:/t', stats.ttest_rel(rho_normal,rho_drusen))

print('')

