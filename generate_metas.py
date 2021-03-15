'''

functions for data analysis of meta/data files


'''


# import matplotlib.pyplot as plt
# import pandas as pd
import ujson as json
import os, fnmatch
# import datetime

from scipy.interpolate import interp1d

import numpy as np 

import matplotlib.pyplot as plt


import time
import pdb
# import math as m

import pdb
import psutil
from joblib import Parallel, delayed
import multiprocessing

import csv

# import scipy.signal

import re
# import random

# from . import hs
from .aws import metas
import multifil as mf 

#generate metas
#========================================================
#phase_sweep
def multi_meta(folder, freq_, lat, phase_, pr, period):
    """
    define a list of frequencies, lattice spacings at 0 strain, phase of activation, and poisson ratio (0=isolattice, .5=isovolumetric)
    returns and saves the metas to 'folder'
    """
   
    meta_list = []
    for l in range(0,len(freq_)):
        for i in range(0,len(lat)):
            for j in range(0,len(phase_)):
                for k in range(0,len(pr)):

                    #static parameters
                    freq = freq_[l]
                    
                    z_line_rest, z_line_amp = 1250, 125  # amplitude is peak to peak, manduca is 10% p2p

                    act_time, act_rise, act_fall, peak_ap = 10, 5, 3, .17
                    #act_time, act_rise, act_fall = 1000*.5/freq, 1000*.1/freq, 1000*.1/freq
                    local_path = folder
                    s3_path =  None

                    t_end = 1000./freq # ms
                    dt = .1 # ms

 
                    # Create time, length, activation traces
                    ls = lat[i]
                    phase = phase_[j]
                    poisson_ratio = pr[k]

                    #write time and z_line traces
                    time_ = np.arange(0,t_end,dt) #in ms
                    activation = metas.actin_permissiveness_workloop(freq, 0, act_time, act_rise, act_fall, time=time_)
                    activation = np.roll(activation, int(round(phase*t_end/dt)))
                    activation = np.divide(activation, np.max(activation))
                    activation = np.multiply(activation, peak_ap) 
                    activation = np.tile(activation,period) 
                    
                    
                    time = np.arange(0,period*t_end,dt) #in ms

                    
                    print('ls=' + str(ls) + ', phase=' + str(phase) + ', poisson_ratio=' + str(poisson_ratio) + ', frequency=' + str(freq))


                    z_line = metas.zline_workloop(z_line_rest, z_line_amp, freq, time)

                    # Emit metafile
                    meta_ = mf.aws.metas.emit(path_local=folder, path_s3=None, time=time, poisson=poisson_ratio, ls=ls, z_line=z_line, actin_permissiveness=activation)
                    meta_list.append(meta_)
    
    return meta_list


def passive_WL_meta(folder, freq_, lat, phase_, pr):
    """
    define a list of frequencies, lattice spacings at 0 strain, phase of activation, and poisson ratio (0=isolattice, .5=isovolumetric)
    returns and saves the metas to 'folder'
    """
    meta_list = []
    for l in range(0,len(freq_)):
        for i in range(0,len(lat)):
            for j in range(0,len(phase_)):
                for k in range(0,len(pr)):

                    #static parameters
                    freq = freq_[l]
                    
                    z_line_rest, z_line_amp = 1250, 125  # amplitude is peak to peak, manduca is 10% p2p

                    act_time, act_rise, act_fall = 10, 5, 5
                    #act_time, act_rise, act_fall = 1000*.5/freq, 1000*.1/freq, 1000*.1/freq
                    local_path = folder
                    s3_path =  None

                    t_end = 1000./freq # ms
                    dt = 2 # ms

                    #write time and z_line traces
                    time = np.arange(0,t_end,dt) #in ms
                    
                    # Create time, length, activation traces
                    ls = lat[i]
                    phase = phase_[j]
                    poisson_ratio = pr[k]

                    print('ls=' + str(ls) + ', phase=' + str(phase) + ', poisson_ratio=' + str(poisson_ratio) + ', frequency=' + str(freq))

                    activation = metas.actin_permissiveness_workloop(freq, phase, act_time, act_rise, act_fall, time)
                    activation = np.multiply(activation, 0) 
                    
                    z_line = metas.zline_workloop(z_line_rest, z_line_amp, freq, time)

                    # Emit metafile
                    meta_ = metas.emit(local_path, s3_path, time, poisson_ratio, ls=ls, z_line=z_line, actin_permissiveness=activation, phase=phase, frequency=freq, z_line_rest=z_line_rest, lat_0 = ls, dt=dt)
                    meta_list.append(meta_)
    return meta_list

#force_velocity
def multi_meta_FV(folder, lat, vel):
    """
    define a list of velocities (in L0/s), +=shortening, -=lengthening, 
    lat = lattice isometric
    returns and saves the metas to 'folder'
    """
    L0=1250
    hold_time = 20 # ms
    t_end = 100. # ms
    dt = 1 # ms
    #write time trace
    time = np.arange(0,t_end,dt) #in ms

    meta_list = []
    for i in range(0,len(lat)):
        for v in range(0,len(vel)):
            velocity = vel[v]
            # Create time, length, activation traces
            ls = lat[i]
            z_line = mf.aws.metas.zline_forcevelocity(L0=L0, hold_time=hold_time, L0_per_sec=velocity, time=time)
            ap=1
            # Emit metafile
            meta_ = mf.aws.metas.emit(path_local=folder, path_s3=None, time=time, poisson=0, ls=ls, z_line=z_line, actin_permissiveness=ap, velocity=velocity, z_line_rest=L0, lat_0 = ls, hold_time=hold_time, dt=dt)
            meta_list.append(meta_)
            print('ls=' + str(ls) + ', velocity=' + str(velocity))

    return meta_list





# wl ############################################
def multi_meta_wl(folder, freq, lat, manduca_sexta, duration, t_in, t_out, peak_ap, phase, period, pr, dt, xb_params):
    """
    define a list of velocities (in L0/s), +=shortening, -=lengthening, 
    lat = lattice isometric
    returns and saves the metas to 'folder'
    """
    ds=1/100
    
    l_amps_ = np.linspace(1250-65, 1250+65, num=8, endpoint=True)
    
    L0 =    1250 #l_amps_[7]    #1250
    z_amp = 125    #125
    
    
    duration = duration # ms
    t_end = 1/freq*1000. # ms
    dt = ds*dt # ms
    #write time trace 
    time_ = np.arange(0,t_end,dt) #in ms
    ap_ = metas.actin_permissiveness_workloop(freq=freq, phase=0, stim_duration=duration, influx_time=t_in, half_life=t_out, time=time_)
    ap_ = np.multiply(peak_ap,np.divide(ap_, np.max(ap_)))
    ap_ = np.subtract(ap_, np.min(ap_))
    
    df = int(1/ds) # .001 for ap, then *100 = > .1 ms
    dt=dt*df
    #pdb.set_trace()
    ap_ = ap_[0::df]

    
    time = np.arange(0,period*t_end,dt)
    ap_ = np.tile(ap_,period)
    
    
    z_line = metas.zline_workloop(L0, z_amp, freq, time)
    
    d10_manduca  = [-0.484075997179581,-0.582775267484216,-0.643556951178390,-0.689553721822144,-0.642618968114505,-0.574812679169490,-0.452429215163164,-0.178144833719841,0.130811831020338,0.446384292241383,0.706374786393262,0.866137415770943,0.925465468319459,0.890717667745911,0.753837387889980,0.574812679169490,0.340954085718920,0.120854626275730,-0.0804974106288299,-0.299715751748096]

    d10_manduca.append(d10_manduca[0])
    real_d10_ph0 =np.add(47.16,d10_manduca)
    
    
    
    d10_to_ls = [i/(3**.5) - 4.5 - 8 for i in real_d10_ph0] # i/(3**.5) - 4.5 - 8
    
    d10_to_ls = np.subtract(d10_to_ls, np.mean([d10_to_ls[5], d10_to_ls[15]]))
    
    plt.plot(d10_to_ls)
    plt.show()
    
    t_def = np.arange(0,41,2)
    
    f = interp1d(t_def, d10_to_ls, kind='cubic', fill_value="extrapolate")
    lat_dt = f(time_[0::df])
    
    lat_dt = np.tile(lat_dt,period)
    # lat_dt = np.subtract(lat_dt, lat_dt[0])
    #plt.plot(lat_dt)
    #plt.show()
    
    #
        

    
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
                
            # plt.plot(ls)
            # plt.show()    
            # # pdb.set_trace()
            
            phi = phase[j]
            ap = np.roll(ap_, int(round(phase[j]*t_end/dt)))

            # ap = np.multiply(peak_ap, np.divide(ap,ap))  # uncomment for tetanus
            # pdb.set_trace()

            #ls = np.sin(2 * np.pi * freq * time/1000) + ls

            plt.plot(time, ap)
            plt.show()
            # Emit metafile
            meta_ = mf.aws.metas.emit(path_local=folder, path_s3=None, time=time, poisson=pr, ls=ls, z_line=z_line, actin_permissiveness=ap, z_line_rest=L0, lat_0 = lat[i], dt=dt, t=time, ap=ap, phase = phase[j], frequency=freq, xb_params = xb_params, manduca_sexta = manduca_sexta)
            meta_list.append(meta_)
            print('ls=' + str(ls))

    return meta_list

############################################





#twitch
def multi_meta_twitch(folder, freq, lat, duration, t_in, t_out, peak_ap):
    """
    define a list of velocities (in L0/s), +=shortening, -=lengthening, 
    lat = lattice isometric
    returns and saves the metas to 'folder'
    """
    L0=1250
    duration = duration # ms
    t_end = 1/freq*1000. # ms
    dt = .001 # ms
    #write time trace 
    time_ = np.arange(0,t_end,dt) #in ms
    ap = metas.actin_permissiveness_workloop(freq=freq, phase=0, stim_duration=duration, influx_time=t_in, half_life=t_out, time=time_)
    ap = np.multiply(peak_ap,np.divide(ap, np.max(ap)))
    ap = np.subtract(ap, np.min(ap))
    dt=dt*100
    ap = ap[0::100]
    
    period=1
    time = np.arange(0,period*t_end,dt)
    ap = np.tile(ap,period)
    
   # pdb.set_trace()
    
    meta_list = []
    for i in range(0,len(lat)):
        # Create time, length, activation traces
        ls = lat[i]
        z_line = L0
        # Emit metafile
        meta_ = mf.aws.metas.emit(path_local=folder, path_s3=None, time=time, poisson=0, ls=ls, z_line=z_line, actin_permissiveness=ap, z_line_rest=L0, lat_0 = ls, dt=dt, t=time, ap=ap)
        meta_list.append(meta_)
        print('ls=' + str(ls))

    return meta_list

#twitch
def multi_meta_tetanus(folder, lat):
    """
    define a list of velocities (in L0/s), +=shortening, -=lengthening, 
    lat = lattice isometric
    returns and saves the metas to 'folder'
    """
    L0=1250
    t_end = 40. # ms
    dt = .1 # ms
    #write time trace 
    time = np.arange(0,t_end,dt) #in ms
    
    meta_list = []
    for i in range(0,len(lat)):
        # Create time, length, activation traces
        ls = lat[i]
        z_line = [L0 for i in time]
        ap = [1 for i in time]
        #pdb.set_trace()
        # Emit metafile
        meta_ = mf.aws.metas.emit(path_local=folder, path_s3=None, time=time, poisson=0, ls=ls, z_line=z_line, actin_permissiveness=ap, z_line_rest=L0, lat_0 = ls, dt=dt, t=time, ap=ap)
        meta_list.append(meta_)
        print('ls=' + str(ls))

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
    
    paths = [wk_dir + '/' + i + '.meta.json' for i in to_do(wk_dir)]
    start = time.time()
    
    
    #num_cores = multiprocessing.cpu_count()-1
    num_cores = psutil.cpu_count(logical=False)
    if num_cores == 10:
        num_cores = 10
    
    Parallel(n_jobs = num_cores)(delayed(run_single)(paths[i]) for i in range(len(paths)))
    end = time.time()
    print('total time: ' + str((end - start)/60) + ' minutes')
    return
    
def run_series(wk_dir):
    paths = [wk_dir + '/'+ i + '.meta.json' for i in to_do(wk_dir)]
    start = time.time()
    [run_single(paths[i]) for i in range(len(paths))]
    end = time.time()
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
    data_dict['ap'] = mf['actin_permissiveness']
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

def FV_meta_to_data_dict(data_name):
    meta_name = data_name[0:-10]+ '.meta.json'
    with open(data_name, 'r') as d:
        df = json.load(d)
    with open(meta_name, 'r') as m:
        mf = json.load(m)  
        
    mNmm2 = 1000 # convert pN/nm2 to mN/mm2
    csa = 5600 # nm2 cross sectional area of the hs unit
    strain = np.divide(mf['z_line'], mf['z_line_rest'])
    stress = np.multiply(df['axial_force'], mNmm2/csa) # units of mN/mm2 for comparison to manduca 
    d_strain = np.diff(np.append(strain, strain[0]))
    
    dz = np.diff(np.append(mf['z_line'], mf['z_line'][0]))
    f = df['axial_force']
    w_ = -np.multiply(f,dz)
    
    work_inc = -np.multiply(stress,d_strain)
    #pdb.set_trace()
    w = np.sum(work_inc)

    out_fns = [
        lambda m: np.multiply(df['timestep'],df['timestep_length']),
        lambda m: mf['name'],
        lambda m: mf['lattice_spacing'],
        lambda m: mf['z_line'],
        lambda m: df['axial_force'],
        lambda m: w
    ]
    out_names = ('time','name', 'lattice_spacing', 'z_line', 'axial_force', 'work_per_cycle')
    
    data_dict = {n:f(m) for f,n in zip(out_fns, out_names)}
    data_dict['lat'] = mf['lat_0']
    data_dict['hold_time'] = mf['hold_time']
    data_dict['dt'] = mf['dt']
    data_dict['strain'] = strain
    data_dict['stress'] = stress
    data_dict['lat'] = mf['lat_0']
    data_dict['velocity'] = mf['velocity']
    data_dict['work_'] = w_
    return data_dict
   
def FV_data_dicts(folder):
    data_files = get_datas(folder)
    data_dicts = [FV_meta_to_data_dict(folder + '/' + i) for i in data_files]
    return data_dicts   
    
def twitch_meta_to_data_dict(data_name):
    meta_name = data_name[0:-10]+ '.meta.json'
    with open(data_name, 'r') as d:
        df = json.load(d)
    with open(meta_name, 'r') as m:
        mf = json.load(m)  
        
    mNmm2 = 1000 # convert pN/nm2 to mN/mm2
    csa = 5600 # nm2 cross sectional area of the hs unit
    strain = np.divide(mf['z_line'], mf['z_line_rest'])
    stress = np.multiply(df['axial_force'], mNmm2/csa) # units of mN/mm2 for comparison to manduca 

    f = df['axial_force']

    out_fns = [
        lambda m: np.multiply(df['timestep'],df['timestep_length']),
        lambda m: mf['name'],
        lambda m: mf['lattice_spacing'],
        lambda m: mf['z_line'],
        lambda m: df['axial_force'],
    ]
    out_names = ('time','name', 'lattice_spacing', 'z_line', 'axial_force', 'work_per_cycle')
    
    data_dict = {n:f(m) for f,n in zip(out_fns, out_names)}
    data_dict['lat_0'] = mf['lat_0']
    data_dict['dt'] = mf['dt']
    data_dict['ap'] = mf['actin_permissiveness']
    data_dict['stress'] = stress
    data_dict['lat'] = mf['lat_0']
    return data_dict
       
def twitch_data_dicts(folder):
    data_files = get_datas(folder)
    data_dicts = [twitch_meta_to_data_dict(folder + '/' + i) for i in data_files]
    return data_dicts



# FV_analysis
#========================================================
def ave_FV(data):
    ht = data['hold_time']
    time = data['time']
    dt = data['dt']
    ave_F = np.mean(data['axial_force'][int(ht/dt):])
    
    V = data['velocity']


    return ave_F







# manage .csv files
def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = os.listdir(path_to_dir)
    return natural_sort([ filename for filename in filenames if filename.endswith( suffix ) ])

def dists_to_array(Filename):
    with open(Filename, 'r') as csvfile:
        so = csv.reader(csvfile, delimiter=',', quotechar='"')
        so_data = []
        for row in so:
            #pdb.set_trace()
            so_data.append(np.double(row))
        so_data = np.asarray(so_data)
        #pdb.set_trace()
        #so_data = np.double(so_data)#.astype('Float64')
        return np.asarray(so_data)

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)    
 
 
 
 
 
 
 