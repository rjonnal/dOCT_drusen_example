import numpy as np
from matplotlib import pyplot as plt
import os,sys,glob
import scipy.optimize as spo

# I used this code to make a figure for my grant proposal, and I'm leaving it here as an example of
# how to interact with the raw/reduced data sets in the ./data directory. Much of what's here is
# inspired by or copied from some of Brennan's code in ../5.original_python_code.

in_directory = os.path.join(os.path.join('data','2018.04.17_processed'),'2018.04.17_IN')
out_directory = os.path.join(os.path.join('data','2018.04.17_processed'),'2018.04.17_OUT')

drusen_data = np.load(os.path.join(out_directory,'2018.04.17_RollAvg_Drusen.npy'))
non_drusen_data = np.load(os.path.join(out_directory,'2018.04.17_RollAvg_NonDrusen.npy'))

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
    plt.plot(x-popt[3],y,'%ss'%c,label=label,markersize=3,alpha=0.5)
    #plt.plot(x,fit,'%s--'%c,label='%s (fit)'%label)
    plt.plot(x-popt[3],fit,'%s--'%'k')
    plt.box(False)
    

ppos = list(range(-5,5))

suffixes = ['isosLoc','isosAng','isosVal','drusDis','img']

plt.figure(figsize=(7.0,5.0))

row_height = 0.2
col_width = 0.33
border = 0.05

bscans = []

for idx,p in enumerate(ppos):
    bottom = (4-(idx%5))*row_height+border/2.0
    left = (idx//5)*0.33+border/2.0
    plt.axes([left,bottom,col_width-border/2.0,row_height-border/2.0])
    pstr = os.path.join(in_directory,'2018.04.17_PPos%0.1f_'%p)
    data = {}
    for s in suffixes:
        f = '%s%s.npy'%(pstr,s)
        data[s]=np.load(f)

    bscan = data['img']
    dd = data['drusDis']
    prof = np.mean(bscan,1)
    pmax = np.argmax(prof)
    z1 = pmax-400
    z2 = pmax+200
    while z2>=bscan.shape[0]:
        z1 = z1 - 1
        z2 = z2 - 1
    bscan = bscan[z1:z2,:]

    
    
    if True:
        bscan = np.log(bscan)
        clim = np.percentile(bscan,(10,99))
    else:
        clim = np.percentile(bscan,(50,99.9))



    xcrop = 25
    plt.imshow(bscan,cmap='gray',interpolation='none',clim=clim,aspect='auto')
    half_turns = p+1.0
    inches = half_turns*0.5/20.0
    mm = inches*25.4
    
    bscans.append((bscan,clim,mm))
        
    
    plt.text(xcrop,0,'%0.1f mm'%mm,color='w',ha='left',va='top',fontsize=9)
    if p==-1 or p==4:
        plt.autoscale(False)

        druse_location = np.where(dd==0)[0]
        plt.plot(np.arange(bscan.shape[1]),data['isosLoc']-z1,'r-')
        plt.plot(np.arange(bscan.shape[1])[druse_location],data['isosLoc'][druse_location]-z1,'b-')

        ang = data['isosAng']

        width = 50
        starts = list(range(50,800,width))
        ends = [s+width for s in starts]
        for s,e in zip(starts,ends):
            mang = np.mean(ang[s:e])
            mid = (s+e)//2
            loc = data['isosLoc'][mid]-z1
            vlen = 50
            slen = 10
            theta = mang/180.0*np.pi*.35
            x1 = mid-np.sin(theta)*slen
            y1 = loc-np.cos(theta)*slen
            x2 = x1-np.sin(theta)*vlen
            y2 = y1-np.cos(theta)*vlen
            #plt.plot([x1,x2],[y1,y2],'y-')
            plt.arrow(x1,y1,x2-x1,y2-y1,color='y',head_width=10,head_length=20,alpha=0.75)
            
        
    plt.xticks([])
    plt.yticks([])
    plt.xlim((xcrop,bscan.shape[1]-xcrop))

idx1 = 4
idx2 = 8

x = 600
wid = 180
xoffset1 = 0
xoffset2 = -15

y = 250
height = 200
yoffset1 = 0
yoffset2 = 110


plt.axes([.66+border/2.0,.66+border/2.0,col_width-border/2.0,.33-border/2.0])
plt.imshow(bscans[idx1][0],cmap='gray',interpolation='none',clim=bscans[idx][1],aspect='auto')
plt.xlim((x+xoffset1,x+wid+xoffset1))
plt.ylim((y+height+yoffset1,y+yoffset1))
plt.xticks([])
plt.yticks([])
plt.text(x+xoffset1,y+yoffset1,'%0.1f mm'%bscans[idx1][2],ha='left',va='top',fontsize=9,color='w')
         
plt.axes([.66+border/2.0,.33+border/2.0,col_width-border/2.0,.33-border/2.0])
plt.imshow(bscans[idx2][0],cmap='gray',interpolation='none',clim=bscans[idx][1],aspect='auto')
plt.xlim((x+xoffset2,x+wid+xoffset2))
plt.ylim((y+height+yoffset2,y+yoffset2))
plt.xticks([])
plt.yticks([])
plt.text(x+xoffset2,y+yoffset2,'%0.1f mm'%bscans[idx2][2],ha='left',va='top',fontsize=9,color='w')


plt.axes([.69,.1,.3,.22])
plot(non_drusen_data,'r','normal')
plot(drusen_data,'b','drusen')
plt.xlabel('$\Theta_{incidence}$ (deg)')
plt.ylabel('reflectance')
plt.yticks([])
plt.legend(fontsize=9,handletextpad=0.005)

plt.savefig('drusen_doct.png',dpi=300)

plt.show()
