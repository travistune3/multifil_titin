

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os, fnmatch
import glob

import multifil as mf
from multifil.aws import metas 

import matplotlib.pyplot as plt
import numpy as np
import winsound
import ujson as json
import shutil
import pandas as pd
from datetime import datetime
from scipy.interpolate import interp1d

import random
import psutil
from joblib import Parallel, delayed
import time

from scipy.signal import butter,filtfilt

import pdb

root_path = 'C:/'


    
#%% 
   
def multi_meta_wl(folder, freq, lat, manduca_sexta, duration, t_in, t_out, peak_ap, phase, period, pr, dt):
    """
    define a list of meta files, each describeing an individual simulation to run
        
    folder: output location of the meta files
    freq: frequency of work loop
    lat: list of lattice spacing offsets
    manduca sexta: 0 uses poisson ratio to updaet lattice spacing at each step, 1 uses data taken from x-ray diffraction during work loops
    duration: on duration for Ca2+
    t_in: Ca2+ influx 
    t_out: Ca2+ decay
    peak_ap: actin permissivenes peak
    phase: phase of activation/stimulation
    period: number of periods to stimulate
    pr: poisson ratio
    dt: time step

    """
    ds=1/100

    L0 =    1250 
    z_amp = 125 # amp here is peak to peak amplitude, 10% of L0
    
    
    duration = duration # ms
    t_end = 1/freq*1000. # ms
    dt = ds*dt # ms
    #write time trace 
    time_ = np.arange(0,t_end,dt) #in ms
    
    # create actin permisiveness time series
    ap_ = metas.actin_permissiveness_workloop(freq=freq, phase=0, stim_duration=duration, influx_time=t_in, half_life=t_out, time=time_)
    ap_ = np.multiply(peak_ap,np.divide(ap_, np.max(ap_)))
    ap_ = np.subtract(ap_, np.min(ap_))
    df = int(1/ds) # .001 for ap, then *100 = > .1 ms
    dt=dt*df
    ap_ = ap_[0::df]
    time = np.arange(0,period*t_end,dt)
    ap_ = np.tile(ap_,period)
    # pdb.set_trace()
    # z line, length of sarcomere
    z_line = metas.zline_workloop(L0, z_amp, freq, time)
    
    # mean d10 from manduca x ray data for 10% peak to peak amplitude at phase of activation of 0, if manduca = 1, this gets interpolated to the time step used and is used instead of poisson ratio to update lattice spacing at each time step
    d10_manduca  = [-0.484075997179581,-0.582775267484216,-0.643556951178390,-0.689553721822144,-0.642618968114505,-0.574812679169490,-0.452429215163164,-0.178144833719841,0.130811831020338,0.446384292241383,0.706374786393262,0.866137415770943,0.925465468319459,0.890717667745911,0.753837387889980,0.574812679169490,0.340954085718920,0.120854626275730,-0.0804974106288299,-0.299715751748096]
    d10_manduca.append(d10_manduca[0])
    real_d10_ph0 =np.add(47.16,d10_manduca)
    
    # convert d10 to ls, the face to face distance of the thick and thin filaments used in the model, uses invertebrate flight muscle for calculation
    d10_to_ls = [i/(3**.5) - 4.5 - 8 for i in real_d10_ph0] # i/(3**.5) - 4.5 - 8
    d10_to_ls = np.subtract(d10_to_ls, np.mean([d10_to_ls[5], d10_to_ls[15]])) # sets 0 as the lattice spacing at strain = 0, not mean lattice, later adds back ls offset for simulation
    
    # plt.plot(d10_to_ls)
    # plt.show()
    
    t_def = np.arange(0,41,2)
    
    f = interp1d(t_def, d10_to_ls, kind='cubic', fill_value="extrapolate")
    lat_dt = f(time_[0::df])
    
    lat_dt = np.tile(lat_dt,period)
    # lat_dt = np.subtract(lat_dt, lat_dt[0])
    #plt.plot(lat_dt)
    #plt.show()

    meta_list = []
    for i in range(0,len(lat)):
        for j in range(0,len(phase)):
            # Create time, length, activation traces
            # ls = lat[i]
            
            #pdb.set_trace()
            
            if manduca_sexta == 1:
                ls = np.add(lat_dt, lat[i])
            elif manduca_sexta == 0:
                ls = lat[i]

            phi = phase[j]
            ap = np.roll(ap_, int(round(phase[j]*t_end/dt)))

            plt.plot(time, ap)
            plt.show()
            # Emit metafile
            meta_ = metas.emit(path_local=folder, path_s3=None, time=time, poisson=pr, ls=ls, z_line=z_line, pca=5, z_line_rest=L0, lat_0 = lat[i], dt=dt, t=time, ap=ap, phase = phase[j], frequency=freq, manduca_sexta = manduca_sexta)
            meta_list.append(meta_)
            # print('ls=' + str(ls))

    return meta_list

# run metas 
#========================================================
def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

def to_do(wk_dir):
    #takes a folder, finds which metas have no data (output) file
    mf_names = fnmatch.filter(os.listdir( wk_dir), '*.meta.json')
    mfb_names = [m[:-10] for m in mf_names]
    df_names = fnmatch.filter(os.listdir(wk_dir), '*.data.json')
    dfb_names = [d[:-10] for d in df_names]
    todo = diff(mfb_names, dfb_names)
    return(todo)

def run_single(path):
    #check if already done
    name = os.path.basename(path)    
    data = path[0:-10]+'.data.json'
    
    running = os.path.split(path)[0] +'/running_' + name[0:-10]
    
    #pdb.set_trace()


    if os.path.isfile(data):
        print('done')
    elif os.path.isfile(running):
        print('running elsewhere')
    else:
        print('running here')
        result = mf.aws.manage(path)        
    #pdb.set_trace()
     
def run_par(wk_dir):
    now_1 = datetime.now()
    dt_string = now_1.strftime("%d/%m/%Y %H:%M:%S")
    print("began = ", dt_string)
    paths = [wk_dir + '/' + i + '.meta.json' for i in to_do(wk_dir)]
    start = time.time()
    #num_cores = multiprocessing.cpu_count()-1
    num_cores = psutil.cpu_count(logical=False)
    Parallel(n_jobs = num_cores)(delayed(run_single)(paths[i]) for i in range(len(paths)))
    end = time.time()
    duration = 250  # milliseconds
    freq = 880  # Hz
    winsound.Beep(freq, duration)
    now_2 = datetime.now()
    dt_string = now_2.strftime("%d/%m/%Y %H:%M:%S")
    print("end = ", dt_string)
    print('total time: ' + str((end - start)/60) + ' minutes')
    return
    
def run_series(wk_dir):
    now_1 = datetime.now()
    dt_string = now_1.strftime("%d/%m/%Y %H:%M:%S")
    print("began = ", dt_string)

    paths = [wk_dir + '/'+ i + '.meta.json' for i in to_do(wk_dir)]
    start = time.time()
    [run_single(paths[i]) for i in range(len(paths))]
    end = time.time()
    duration = 250  # milliseconds
    freq = 880  # Hz
    winsound.Beep(freq, duration)
    now_2 = datetime.now()
    dt_string = now_2.strftime("%d/%m/%Y %H:%M:%S")
    print("end = ", dt_string)
    print('total time: ' + str((end - start)/60) + ' minutes')
    return
    
# data files
#========================================================
def get_datas(wk_dir):
    df_names = fnmatch.filter(os.listdir(wk_dir), '*.data.json')
    return df_names

def get_metas(wk_dir):
    mf_names = fnmatch.filter(os.listdir(wk_dir), '*.meta.json')
    return mf_names

def phase_ave(data, periods, period_length):
    data = np.array(data).reshape(periods, period_length)
    data = data.mean(axis=0)
    return data

def WL_meta_to_data_dict(data_name):
    meta_name = data_name[0:-10]+ '.meta.json'
    with open(data_name, 'r') as d:
        df = json.load(d)
    with open(meta_name, 'r') as m:
        mf = json.load(m)  
    
    mNmm2 = 1000 # convert pN/nm2 to mN/mm2
    csa = 5700 # nm2 cross sectional area of the hs unit
    strain = np.divide(mf['z_line'], mf['z_line_rest'])

    stress = np.multiply(df['axial_force'], mNmm2/csa) # units of mN/mm2 for comparison to manduca 
    d_strain = np.diff(np.append(strain, strain[0]))
    work_inc = -np.multiply(stress,d_strain)
    w = np.sum(work_inc)
    
    dz = np.diff(np.append(mf['z_line'], mf['z_line'][0]))
    f = df['axial_force']
    w_ = -np.multiply(f,dz)
    n = int(mf['timestep_number']*mf['timestep_length']*mf['frequency']/1000 )
    s_rate = 1/mf['timestep_length']
    # print(n)
    sig_length = int(1000/mf['frequency']/mf['timestep_length'])

    # assert n==mf['period']
    
    out_fns = [
        lambda m: np.multiply(df['timestep'][0:int(1000/mf['frequency']/mf['timestep_length'])],df['timestep_length']),
        lambda m: mf['name'],
        lambda m: mf['lattice_spacing'],
        lambda m: mf['z_line'],
        lambda m: w/n
    ]
    out_names = ('time','name', 'lattice_spacing', 'z_line', 'work_per_cycle')
    data_dict = {n:f(m) for f,n in zip(out_fns, out_names)}  
    
    data_dict['poisson_ratio'] = mf['poisson_ratio']
    data_dict['phase'] = mf['phase']
    data_dict['lat'] = mf['lat_0']
    data_dict['strain'] = strain
    data_dict['stress'] = stress
    data_dict['lat_0'] = mf['lat_0']
    data_dict['phase'] = mf['phase']
    data_dict['ap'] = mf['pca']
    data_dict['z_line_rest'] = mf['z_line_rest']
    data_dict['work_inc'] = work_inc
    data_dict['n_periods'] = n
    data_dict['s_rate'] = s_rate*1000
    
    data_dict.update(df)
    
    #phase average 
    for k,v in data_dict.items():
        try:
            data_dict[k] = phase_ave(data_dict[k], n, sig_length)
        except:
            pass
            #print(k)
    # pdb.set_trace()
    return data_dict
           
def WL_data_dicts(folder):
    data_files = get_datas(folder)
    data_dicts = [WL_meta_to_data_dict(folder + '/' + i) for i in data_files]
    return data_dicts



#%% define where meta files are put directory
####################################################

wk_dir = root_path + '/test2'
# 2sxb_with_titin
if not os.path.exists(wk_dir):
    os.mkdir(wk_dir)
    print(wk_dir)

d10_manduca = [-0.484075997179581,-0.582775267484216,-0.643556951178390,-0.689553721822144,-0.642618968114505,-0.574812679169490,-0.452429215163164,-0.178144833719841,0.130811831020338,0.446384292241383,0.706374786393262,0.866137415770943,0.925465468319459,0.890717667745911,0.753837387889980,0.574812679169490,0.340954085718920,0.120854626275730,-0.0804974106288299,-0.299715751748096]
real_d10_ph0 =np.add(47.16,d10_manduca)
d10_to_ls = [i/(3**.5) - 4.5 - 8 for i in real_d10_ph0] # i/(3**.5) - 4.5 - 8
    
#%% 

#clear meta folder
files = glob.glob(wk_dir + '/*')
for f in files:
    os.remove(f)  
    
    
#%% 
 
print(wk_dir)

# name a single frequency, and list of lattice spacings and phases of activations
freq = 25
lat = np.linspace(15,16,1,endpoint=True).tolist()#[15]
phase = np.linspace(.00,1.,1,endpoint=False).tolist()#[0,.1,.2,.3,.4,.5,.6,.7,.8,.9]

# poisson ratio, 0 = constant lattice spacing, .5 = isovolumetric constraint
pr=0.0

# ms=0 means use poisson ratio from above to update lattice spacgin at each time step, ms=1 means overwrite poisson ratio and use time series from x-ray data, 
ms=0

#duration of simulation in periods
period = 1

# time step in ms
dt=.1   # .1
 
# creates the metas based on the above, outputs to wk_dir 
meta_list = multi_meta_wl(wk_dir, freq=freq, lat=lat, manduca_sexta=ms, duration=5, t_in=6, \
                                 t_out=2, peak_ap = .2, phase = phase, period=period, pr=pr, dt=dt)


# for i in range(0, len(meta_list)):
#     plt.plot(meta_list[i]['t'][0:399], np.transpose(meta_list[i]['ap'][0:399]), label = meta_list[i]['lat_0'])
# plt.legend()
# plt.show()

# try:
#     for i in range(0, len(meta_list)):
#         plt.plot(meta_list[i]['t'][0:399], np.transpose(meta_list[i]['lattice_spacing'][0:399]))
#     plt.legend()
#     plt.show()
# except:
#     pass

# for i in range(0, len(meta_list)):
#     plt.plot(meta_list[i]['t'][0:399], np.transpose(meta_list[i]['z_line'][0:399]))
# plt.legend()
# plt.show()

# print(len(meta_list[0]['t']))

# print(len(meta_list))


 #%% run metas


print(wk_dir)

# run meta files in parralelelaleé
# run_par(wk_dir) 

# run in series
run_series(wk_dir) 


#%% plots stuff
i=1

# wk_dir = wk_dir_list[i]

# print(wk_dir_list[i][100:])

# data_as_Json(wk_dir)


mean_work = (1.2236,0.0909,-0.8567,-1.4888,-2.1519,-2.4017,-2.0171,-0.2661,1.8313,1.9622)
work_std = (0.2584,0.1839,0.3612,0.6683,0.8867,0.8204,0.4848,0.2905,0.6692,0.3493)
data = WL_data_dicts(wk_dir)
data[0].keys()

# plots each work loop (phase averaged)
for i in range(0,len(data)):
    plt.plot(np.multiply(np.asarray(data[i]['strain']),2), np.transpose(data[i]['stress']), label = data[i]['phase'])
    plt.legend()
plt.title('work loops')
plt.show()

# plots phase of activation vs net work, and compares with manduca sexta experimental data
plt.plot([i['phase'] for i in data], [np.sum(i['work_per_cycle']) for i in data], '.')
plt.errorbar(np.arange(0,1,.1), mean_work, work_std, label='in vivo - argonne')
# plt.ylim((-6,3))
plt.savefig(wk_dir + '/W_v_phase.eps', format='eps')
plt.title('phase of activation vs net work')
plt.show()

plt.plot(data[0]['time'],np.transpose([i['ap'] for i in data]))
plt.title('activation')
plt.show()

# lattice offset (L_0) vs net work 
for i in range(0,len(data)):
    plt.plot(data[i]['lat_0'], data[i]['work_per_cycle'],'.', label = data[i]['lat_0'])
# plt.legend()
# plt.ylim((-5,3))
plt.savefig(wk_dir + '/w_v_lat.eps', format='eps')
plt.title('lattice vs net work ')
plt.show()

# plots each work loop (phase averaged)
for i in range(0,len(data)):
    plt.plot(np.multiply(np.asarray(data[i]['time']),2), np.transpose(data[i]['stress']), label = data[i]['phase'])
    plt.legend()
plt.title('stress vs time')
plt.show()






# i=0
# plt.plot(data[i]['time'], 100*np.transpose(data[i]['xb_trans_12']), label = (data[i]['lat_0']+8+4.5)*3**.5)
# plt.plot(data[i]['time'], 100*np.transpose(data[i]['xb_trans_23']), label = (data[i]['lat_0']+8+4.5)*3**.5)
# plt.plot(data[i]['time'], 100*np.transpose(data[i]['xb_trans_31']), label = (data[i]['lat_0']+8+4.5)*3**.5)
# # plt.plot(data[i]['time'], np.transpose(data[i]['stress']), label = (data[i]['lat_0']+8+4.5)*3**.5)
# plt.legend()
# plt.ylim((0,1600))
# plt.show()

i=0
plt.plot(data[i]['time'], 100*np.transpose(data[i]['xb_fraction_loose']), label = 'loose')
plt.plot(data[i]['time'], 100*np.transpose(data[i]['xb_fraction_tight']), label = 'tight')
# plt.plot(data[i]['time'], np.transpose(data[i]['stress']), label = (data[i]['lat_0']+8+4.5)*3**.5)
plt.legend()
plt.ylim((0,100))
plt.title('xb fraction in loose and tight states')
plt.show()


#   invert flight
#   (ls + 8 + 4.5)*(3**.5) = d10
#   d10/(3**.5) - 4.5 - 8  = ls



#   vertebrate
#   (ls + 8 + 4.5)*(3/2) = d10
#   d10/(3/2) - 4.5 - 8 = ls





# i=5
# plt.plot(data[i]['time'], np.transpose(data[i]['stress']), label = data[i]['lat_0'])
# plt.legend()
# plt.ylim((-5,60))
# plt.savefig(wk_dir + '/stress.eps', format='eps')
# plt.show()

# i=7
# plt.plot(data[i]['time'], np.transpose(data[i]['stress']), label = data[i]['lat_0'])
# plt.legend()
# plt.ylim((-5,60))
# plt.show()

# i=3
# plt.plot(data[i]['strain'], np.transpose(data[i]['stress']), label = data[i]['phase'])
# plt.legend()
# plt.show()

# for i in range(0,len(data)):
#     plt.plot(data[i]['time'], np.transpose(data[i]['ap']), label = data[i]['phase'])
# plt.legend()
# plt.show()

# i=0
# plt.plot(data[i]['time'], np.transpose(data[i]['stress']), label = data[i]['phase'])
# plt.plot(data[i]['time'],np.multiply(40, np.transpose(data[i]['ap'])), label = data[i]['phase'])
# plt.legend()
# plt.show()

#%%

# for running a list of folders, each with separate conditions

random.shuffle(wk_dir_list)


for i in range(0, len(wk_dir_list)):
    
    print(wk_dir_list[i] + '\n')
    mf.tt.run_par(wk_dir_list[i]) 
    time.sleep(2) 
    try:
        files = glob.glob('D:/Travis/tmp/*')
        for f in files:
            shutil.rmtree(f)
    except:
        pass
    try:
        files = glob.glob('F:/tmp/*')
        for f in files:
            shutil.rmtree(f)
    except:
        pass
        # F:\tmp
    time.sleep(2)      


