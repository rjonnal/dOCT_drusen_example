## Using directional OCT to understand photoreceptor visibility in AMD

### Introduction

Investigators using adaptive optics (AO) retinal imaging instruments have reported reduced visibility of the cone photoreceptors overlying drusen as compared to those in unaffected portions of the same retinae or those in healthy retinae. Two compatible hypotheses have been offered to explain this phenomenon. Some have suggested that disease-related deformation of the photoreceptor outer segment (OS) reduces its ability to act as a wave guide, thus reducing the cell's familiar reflectance pattern. Others have suggested that drusen disorient the photoreceptors away from the eye's pupil, thus reducing the amount of reflected light that can be detected outside the eye.

In order to assess the contributions of these two potential factors to reduced photoreceptor visibility, we employed a custom research-grade OCT optical coherence tomography (OCT) system, along with a directional experimental protocol to acquire OCT images at a variety of positions in the pupil.

This repository is intended to illustrate the analytical approach that we employed.

### Methods

Directional OCT (dOCT) was realized by translating the OCT imaging beam across the horizontal diameter of the pupil. As described below, our analytical approach did not require knowledge of the beam's position in the pupil, so we acquired images from one edge of the pupil to the other in increments of 0.64 mm. At each location, between 1200 and 1600 B-scans were acquired. These were aligned and averaged using a custom semi-rigid body algorithm. For each pupil location, at least 50 B-scans were averaged to generate a single averaged image. Pupil locations were omitted from analysis if 50 B-scans could not be identified with sufficiently high correlation (e.g., when there was too much eye movement normal to the plane of the B-scan). In the resulting average images, a semi-automated method was used to segment the inner-outer segment (IS/OS) band and fit it with a smooth curve. (The cone outer segment (COST) band was not studied because a clear boundary between COST and retinal pigmented epithelium (RPE) was rarely observed above drusen, and contamination by light scattered from RPE was unavoidable.) 

At each A-scan of the average B-scan, the tangent to the IS/OS curve and the IS/OS amplitude (integrated over three pixels) was recorded. To account for the potentially confounding variable of overall image brightness (as a function of pupil position), the IS/OS amplitude was normalized by the amplitude of an overlying region in the outer nuclear layer (ONL), which was separately shown to have low directional dependence. Thus each amplitude value recorded represents the factor by which the IS/OS was more reflective than the corresponding ONL (in linear scale).

In addition, in each average B-scan, the boundaries of the drusen were determined visually, and each A-scan was labeled as 'drusen', 'non-drusen', or 'transitional'. Transitional zones were omitted from further analysis.

Thus for each pupil entry position, an **averaged B-scan** was recorded, along with records of the following parameters, all functions of lateral position in the B-scan (i.e., A-scan location):

* IS/OS axial location
* IS/OS angle
* IS/OS amplitude
* distance to druse margin

Pupil entry position was not monitored, but the beam was stepped across the pupil. Images and quantification were labeled arbitrarily between integers $-M$ and $N$, with positive values designated with 'p' and negative values designated with 'n'.

#### Visualizing data from one pupil position

The following example illustrates the data recorded from a single pupil entry position.


```python
import numpy as np
from matplotlib import pyplot as plt
import os
import scipy.optimize as spo
import scipy.stats as sps

plt.style.use('seaborn-deep')
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
subject_markers = ['s','o','d','^']
print_dpi = 300
screen_dpi = 100
figure_size = (5,4)
pupil_position = 'p01'
subject_folder = './data/subject_02/'

# load B-scan and convert to dB
bscan_fn = os.path.join(subject_folder,'pupil_position_%s_average_bscan.npy'%pupil_position)
bscan = np.load(bscan_fn)
dB = 20*np.log10(bscan)
clim = (55,85)

# load IS/OS location
isos_location_fn = os.path.join(subject_folder,'pupil_position_%s_isos_axial_location.npy'%pupil_position)
isos_location = np.load(isos_location_fn)
x = np.arange(len(isos_location))

# load IS/OS angle
isos_angle_fn = os.path.join(subject_folder,'pupil_position_%s_isos_angle.npy'%pupil_position)
isos_angle = np.load(isos_angle_fn)

# load IS/OS amplitude
isos_amplitude_fn = os.path.join(subject_folder,'pupil_position_%s_isos_amplitude.npy'%pupil_position)
isos_amplitude = np.load(isos_amplitude_fn)

# show the B-scan and resulting IS/OS curve
plt.figure(figsize=figure_size,dpi=screen_dpi)
plt.imshow(dB,clim=clim,cmap='gray')
plt.ylim((2000,1300))
plt.xticks([])
#plt.yticks([])
plt.colorbar()
plt.plot(x,isos_location,label='IS/OS location',alpha=0.5)
plt.legend()
plt.savefig('figures/bscan_with_trace.png',dpi=print_dpi)

# plot angle and amplitude as functions of x
plt.figure(figsize=figure_size,dpi=screen_dpi)
plt.plot(x,isos_angle,label='IS/OS angle')
plt.plot(x,20*np.log10(isos_amplitude),label='IS/OS amplitude (dB)')
plt.legend()
plt.xlabel('x location')
plt.savefig('figures/amplitude_vs_location.png',dpi=print_dpi)
```


![png](dOCT_drusen_example_files/dOCT_drusen_example_1_0.png)



![png](dOCT_drusen_example_files/dOCT_drusen_example_1_1.png)


#### Data aggregated over pupil positions

Data associated with the pupil-position-specific B-scans were aggregated, for drusen and non-drusen regions of the images separately. For each subject, these are stored in two files: 

* ```directionality_raw_data_drusen_centered.npy```
* ```directionality_raw_data_nondrusen_centered.npy```

Each contains an Nx2 matrix, where N represents the number of data points aggregated from all pupil positions. The two columns represent the IS/OS curve angle and normalized amplitude. Because the displacement of the Stiles-Crawford peak from the pupil center is not of interest, and because the pupil position was not tracked, angles for each subject were zero-centered by subtracting the average angle.

#### Visualization of aggregated data

The plot below illustrates the data aggregated from one subject's non-drusen IS/OS. In order to calculate the angle correctly, the x and z sampling densities of the OCT image must be considered. The width of the OCT scan on the retina was $2 mm$, sampled with 800 A-scans, resulting in a pixel width of $2.5 \mu m$. The depth corresponding to a single pixel was $5 \mu m$, which requires a scaling of the angle by a factor of 2. Assuming an eye focal length $f$ of $16.7 mm$, and a tangent angle $\theta$, the effective pupil position $x$ is given by:

$$x = \tan(\theta)f$$




```python
def filename_to_x_mm_amplitude(fn):
    angle_amplitude = np.load(fn)
    # divide the angle by 2 to account for x-z sampling anisotropy
    angle = angle_amplitude[:,0]/2.0
    angle = (angle/180*np.pi)
    x_mm = np.tan(angle)*16.67
    amplitude = angle_amplitude[:,1]
    return x_mm,amplitude

nondrusen_aggregate_fn = os.path.join(subject_folder,'directionality_raw_data_nondrusen_centered.npy')
x_mm,amplitude = filename_to_x_mm_amplitude(nondrusen_aggregate_fn)

plt.figure(figsize=figure_size,dpi=screen_dpi)
plt.plot(x_mm,amplitude,'.',alpha=0.2,markersize=1)
plt.gca().set_ylim(bottom=0)
plt.xlim((-2.5,2.5))

plt.ylabel('normalized amplitude (ADU)')
plt.xlabel('effective pupil position (mm)')
plt.savefig('figures/amplitude_vs_angle.png',dpi=print_dpi)
plt.show()

```


![png](dOCT_drusen_example_files/dOCT_drusen_example_3_0.png)


#### Improving data visualization using rolling averaging

While a clear dependence of IS/OS amplitude on angle of illumination is visible in the figure above, substantial variance at all angles makes the relationship difficult to appreciate visually. To improve visualization, a rolling average technique was employed. A window of width $5.0^\circ$ was stepped across the range of angles present in a subject's measurements, in increments of $2.0^\circ$, and the average amplitude and standard deviation of amplitude were recorded at each location. The result of this averaging is shown below.


```python
def rolling_median(pupil_position,amp,window_width=2.0,step_size=1.0,diagnostics=False):
    
    # find the start and end angles
    t_start = np.min(pupil_position)
    t_end = np.max(pupil_position)
    #print(t_start,t_end)
    window_centers = []
    amplitude_mean = []
    amplitude_median = []
    amplitude_std = []

    for t in np.arange(t_start,t_end+step_size,step_size):

        # find he indices in the angle array where the angle falls in our window
        idx = np.where(np.logical_and(pupil_position>=t,pupil_position<t+window_width))[0]

        if len(idx)>0:
            window_centers.append(t+window_width/2.0)        
            amplitude_mean.append(np.mean(amp[idx]))
            amplitude_median.append(np.median(amp[idx]))
            amplitude_std.append(np.std(amp[idx]))



    return np.array(window_centers),np.array(amplitude_mean),np.array(amplitude_median),np.array(amplitude_std)

window_centers, amplitude_mean, amplitude_median, amplitude_std = rolling_median(x_mm,amplitude)
plt.figure(figsize=figure_size,dpi=screen_dpi)
# plot the raw data with low alpha
plt.plot(x_mm,amplitude,'.',alpha=0.1,markersize=1,label='raw data')
# plot the average with standard deviation bars
plt.errorbar(window_centers,amplitude_mean,amplitude_std,label='rolling mean')
plt.errorbar(window_centers,amplitude_median,amplitude_std,label='rolling median')
plt.xlabel('effective pupil position (mm)')
plt.ylabel('normalized amplitude (ADU)')
plt.gca().set_ylim(bottom=0)
plt.xlim((-2.5,2.5))
plt.legend()
plt.savefig('figures/rolling_median.png',dpi=print_dpi)
plt.show()

```


![png](dOCT_drusen_example_files/dOCT_drusen_example_5_0.png)


#### A model for directionality

The amplitude distribution in the pupil of light backscattered by the retina is typically described as a sum of a constant (diffuse) component and a Gaussian (directional) component:

$$a(x,y) = B+A\times 10^{-\rho[(x-x_0)^2+(y-y_0)^2]},$$

where $B$ represents the diffuse component, $A$ represents the height of the Gaussian component, $\rho$ represents the directionality coefficient, $x$ ($y$) represents the horizontal (vertical) position in the pupil, and $x_0$ ($y_0$) the location of the Stiles-Crawford peak. $x$, $x_0$, $y$, and $y_0$ are typically expressed in $mm$, which results in $\rho$ having units of $mm^{-2}$. If, as in the present work, only the horizontal dimension of the directionality function is modeled, the equation simplifies to:

$$a(x) = B+A\times 10^{-\rho[(x-x_0)^2]}$$

The width of Gaussian distributions is more commonly described by standard deviation $\sigma$, and from that definition it is apparent that $\rho=\frac{1}{2\sigma^2}$, and that wider distributions (higher $\sigma$) have lower directionality coefficients $\rho$, and narrower ones have higher $\rho$.

#### Estimating $B$, $A$, and $\rho$

To estimate these three parameters, the smoothed functions described above were fit with a four parameter model. ($x_0$ was used to optimize the fits, but was not considered in further analysis.) In the next step, the median-filtered data from one subject's non-drusen and drusen regions are fit separately, and the results of the fits visualized side by side.


```python
def gaussian(x_mm, B, A, rho, x0_mm):
    return B + A*(10**(-rho*(x_mm-x0_mm)**2))

amax = None

def fit_data_from_file(fn):
    x_mm,amp = filename_to_x_mm_amplitude(fn)
    wc,amean,amed,std = rolling_median(x_mm,amp)
    fit_params = spo.curve_fit(gaussian,wc,amed)[0]
    return fit_params, x_mm, amp, wc, amed, std

subject_folder='data/subject_04'

plt.figure(figsize=(figure_size[0]*2,figure_size[1]),dpi=screen_dpi)
titles = ['non-drusen','drusen']

for idx,fn in enumerate(['directionality_raw_data_nondrusen_centered.npy','directionality_raw_data_drusen_centered.npy']):
    ffn = os.path.join(subject_folder,fn)
    plt.subplot(1,2,idx+1)
    fit_params, x_mm, amp, wc, amed, std = fit_data_from_file(ffn)
    if idx==0:
        amax = np.max(amp)
    afit = gaussian(wc,*fit_params)
    plt.plot(x_mm,amp,'.',alpha=0.1,markersize=1,label='raw data')
    # plot the average with standard deviation bars
    plt.errorbar(wc,amed,std,label='rolling median')
    plt.plot(wc,afit,'k--',label='fit')
    plt.xlabel('effective pupil position (mm)')
    plt.ylabel('normalized amplitude (ADU)')
    plt.gca().set_ylim(bottom=0)
    plt.gca().set_ylim(top=amax)
    plt.xlim((-2.5,2.5))
    plt.legend()
    plt.title(titles[idx])
    
plt.savefig('figures/fitting_normal_drusen.png',dpi=print_dpi)
```

    <ipython-input-4-df347e4ee221>:2: RuntimeWarning: overflow encountered in power
      return B + A*(10**(-rho*(x_mm-x0_mm)**2))



![png](dOCT_drusen_example_files/dOCT_drusen_example_9_1.png)


#### Comparing $B$, $A$, and $\rho$ between non-drusen and drusen regions



```python
subject_folders = ['data/subject_01', 'data/subject_02', 'data/subject_03', 'data/subject_04']
drusen_labels = ['non-drusen','drusen']

# positions to visualize fits:
x_mm_fit = np.arange(-2.5,2.55,0.05)

linestyles = ['-',':']
marker_alpha = 0.75

B_grid = np.zeros((len(subject_folders),len(drusen_labels)))
A_grid = np.zeros((len(subject_folders),len(drusen_labels)))
rho_grid = np.zeros((len(subject_folders),len(drusen_labels)))
plt.figure(figsize=(figure_size[0]*.75,figure_size[1]),dpi=screen_dpi)

for subject_idx,subject_folder in enumerate(subject_folders):
    subject_label = os.path.split(subject_folder)[1].replace('_',' ')
    for drusen_idx,fn in enumerate(['directionality_raw_data_nondrusen_centered.npy','directionality_raw_data_drusen_centered.npy']):
        drusen_label = drusen_labels[drusen_idx]
        full_fn = os.path.join(subject_folder,fn)
        fit_params, x_mm, amp, wc, amed, std = fit_data_from_file(full_fn)
        B, A, rho, x0_mm = fit_params
        y_fit = gaussian(x_mm_fit, B, A, rho, 0.0)
        plt.plot(x_mm_fit,y_fit,marker=subject_markers[subject_idx],alpha=marker_alpha,markevery=20,color=color_cycle[drusen_idx],linestyle=linestyles[drusen_idx], label='%s, %s'%(subject_label,drusen_label))

        B_grid[subject_idx,drusen_idx] = B
        A_grid[subject_idx,drusen_idx] = A
        rho_grid[subject_idx,drusen_idx] = rho
        
plt.gca().set_ylim(bottom=0)
plt.xlim((-2.5,2.5))
plt.xlabel('effective pupil position (mm)')
plt.ylabel('amplitude fit (ADU)')

        
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


```

    <ipython-input-4-df347e4ee221>:2: RuntimeWarning: overflow encountered in power
      return B + A*(10**(-rho*(x_mm-x0_mm)**2))
    <ipython-input-4-df347e4ee221>:2: RuntimeWarning: overflow encountered in power
      return B + A*(10**(-rho*(x_mm-x0_mm)**2))
    <ipython-input-4-df347e4ee221>:2: RuntimeWarning: overflow encountered in power
      return B + A*(10**(-rho*(x_mm-x0_mm)**2))
    <ipython-input-4-df347e4ee221>:2: RuntimeWarning: overflow encountered in power
      return B + A*(10**(-rho*(x_mm-x0_mm)**2))



![png](dOCT_drusen_example_files/dOCT_drusen_example_11_1.png)



```python
grids = [A_grid,B_grid,rho_grid]
labels = ['A','B',r'$\rho$']
xtl = ['nondrusen','drusen']

plt.figure(figsize=(figure_size[0]*2.5,figure_size[1]),dpi=screen_dpi)
for idx,(grid,label) in enumerate(zip(grids,labels)):

    tres = sps.ttest_rel(grid[:,0],grid[:,1])
    p = tres.pvalue
    print(p)

    if label=='B':
        # do a second t-test excluding the outlier
        tres = sps.ttest_rel(grid[:3,0],grid[:3,1])
        p_nooutlier = tres.pvalue
        print(p_nooutlier)

    plt.subplot(1,3,idx+1)
    plt.boxplot(grid,labels=xtl,notch=False)
    
    for idx in range(len(subject_folders)):
        plt.plot(1.2,grid[idx,0],subject_markers[idx],color=color_cycle[0],alpha=markeralpha)
        plt.plot(2.2,grid[idx,1],subject_markers[idx],color=color_cycle[1],alpha=markeralpha)
        
    plt.title('parameter %s, p=%0.3f'%(label,p))
    
    
plt.show()
print(B_grid)
```

    0.06478700447883119
    0.31015640967256397
    0.07480797749210936
    0.029984353846298824



![png](dOCT_drusen_example_files/dOCT_drusen_example_12_1.png)


    [[1.90206923 1.1023884 ]
     [1.8201435  1.29117264]
     [1.41914991 1.15342948]
     [1.62376128 1.99583339]]



