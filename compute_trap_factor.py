
from trap_metric import TrapMetric
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

def convert_time_interval(time,conc,time_int=0.1):
    start_time=time[0]
    time_array = []
    conc_array = []
    for i in range(len(time)):
        new_time=time[i]
        ts = new_time/start_time
        if ts>=time_int:
            time_array.append(time[i])
            conc_array.append(conc[i])
            start_time=new_time
    return(time_array,conc_array)


def clean_data(time,l_grad,thresh_freq=1,bin_num=50):
    data=np.histogram(l_grad,bins=bin_num)
    #print(data[0])
    # print(data)
    flag=False
    count=0
    bin_val_min=0
    bin_val_max=0
    for i in range(len(data[0])):
        if data[0][i] >=10 and not flag:
            flag=True
            count+=1
            bin_val_min = data[1][i]
        elif data[0][i] <=1 and flag:
            count+=1
            bin_val_max=data[1][i]
            break

    mask_out = (l_grad <= bin_val_max) & (l_grad >= bin_val_min)
    new_time = np.array(time)[mask_out]
    l_grad_new = l_grad[mask_out]

    return(new_time,l_grad_new)




def ComputeTrapFactor(sim,plot=False):
    trap_met = TrapMetric(sim)

    time_arr = np.array(sim.steps)
    
    complx_conc = np.array(sim.observables[len(sim.observables)-1][1])
    sel_time = (time_arr >= 1e-3)
    sel_indx = np.argwhere(sel_time)[0][0]

    filter_time,filter_conc = convert_time_interval(time_arr[sel_indx-1:],complx_conc[sel_indx-1:],time_int=1.3)
    final_time = np.concatenate((time_arr[:sel_indx],filter_time[:]))
    final_conc = np.concatenate((complx_conc[:sel_indx],filter_conc[:]))

    #Uncleaned version - GRAD1
    l_grad_unclean = trap_met.calc_slope(final_time,final_conc,mode='log')

    clean_time1,l_grad = clean_data(final_time,l_grad_unclean)

    actual_l_grad = l_grad_unclean
    actual_time = final_time
    
    #Finding time points by just visual picking
    first_peak_mask = actual_time<=1
    first_peak_indx = np.argmax(actual_l_grad[first_peak_mask])
    first_peak = actual_time[first_peak_mask][first_peak_indx]

    second_regime_mask = (actual_time>=1)
    #second_peak_indx = np.argmax(actual_l_grad[second_regime_mask])
    second_peak_indx,_ = find_peaks(actual_l_grad[second_regime_mask],width=1,prominence=1)
    
    if len(second_peak_indx)==0:
        #if second_peak_indx==0:
        second_peak=first_peak
        #else: 
        #second_peak = actual_time[second_regime_mask][second_peak_indx]
    else:     
        #if second_peak_indx[-1]==0:
        #    second_peak=first_peak
        #else: 
        second_peak = actual_time[second_regime_mask][second_peak_indx[-1]]
    
    time_bounds = [first_peak,second_peak]

    #print(time_bounds)
    lag_time = np.log(time_bounds[1]/time_bounds[0])
    tf=time_bounds[1]/time_bounds[0]
    #print("Lag Factor: ",lag_time)

    #min_conc = np.argmin(final_conc[second_regime_mask])
    #mask_int = (actual_time>20) & (actual_time<1e4)
    #print("Trapped Yield: ",final_conc[second_regime_mask][min_conc]/10)
    #print("Avg trapped yield: ",np.mean(final_conc[mask_int])/10)

    #print("Trapped Yield: ",final_conc[valley_mask][min_grad]/1000) 
    
    if plot==True:
        fig,[ax1,ax2] = plt.subplots(1,2,figsize=(15,4))
        ax1.plot(final_time,final_conc)
        ax1.vlines(time_bounds[0],ymin=0,ymax=100,color='k',linestyle='--')
        ax1.vlines(time_bounds[1],ymin=0,ymax=100,color='r',linestyle='--')
        ax1.set_xscale("log")
        ax1.set_xlabel('time [s]')
        ax1.set_title('Yield concentation')


        #ax2.plot(final_time[first_peak_mask],l_grad_unclean[first_peak_mask],alpha=0.6,marker='o')
        #ax2.plot(final_time[second_regime_mask],l_grad_unclean[second_regime_mask],alpha=0.6,marker='o',color='orange')
        ax2.plot(final_time[:],l_grad_unclean[:],alpha=0.6,marker='o')
        ax2.vlines(time_bounds[0],ymin=0,ymax=max(l_grad_unclean),color='k',linestyle='--')
        ax2.vlines(time_bounds[1],ymin=0,ymax=max(l_grad_unclean),color='r',linestyle='--')
        ax2.set_xscale("log")
        ax2.set_xlabel('time [s]')
        ax2.set_title('Gradient')
        
    return tf

