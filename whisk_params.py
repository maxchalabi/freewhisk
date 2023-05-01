import sys
import numpy as np
import pandas as pd

import whisk_tools, misc_tools


def extract_whisking_envelope(whisk_arr, prominence=0.03, cutoff=4, pixel_scale=100, bin_size=3, mean_setpoint = 60):
    """
    This function reunites some of the functions in whisk_tools into a single function
    to extract the global motion of the whiskerpad from the raw whisker pixel data.

    Parameters
    ----------
    whisk_arr : 150 x 500 numpy array
        raw whisker pixel data
    prominence : float, default 0.03
        prominence parameter for scipy.signal.find_peaks
    pixel_scale : float, default 100
        defines by how much pixel values will be scaled to weight their importance in comparison to y-position when tracking peaks.
    bin_size : int, default 3
        defines the size of the bins used to compute the mean movement of peaks over time with sliding windows in whisk_arr.
    mean_setpoint : float, default 60
        defines the starting point for the whisking envelope.
        Currently working on a way to find the true absolute whisking setpoint.
    
    Returns
    -------
    mean_angle : list
        y coordinates to show global motion of whiskerpad

    """
    
    peak_coords, raw_img = whisk_tools.collect_peaks(whisk_arr, prominence=prominence, cutoff=cutoff, pixel_scale=pixel_scale)
    thresh = whisk_tools.peak_thresh(peak_coords)
    mmp_slices = whisk_tools.track_mmp_slices(peak_coords, thresh, slice_size=bin_size)
    patched_mmp_slices = whisk_tools.patch_mmp_slices(mmp_slices, mean_setpoint, raw_img)
    mean_angle = whisk_tools.average_mmp_slices(patched_mmp_slices, (0, np.shape(whisk_arr)[1]))
    
    return mean_angle


def compute_trial_amps_and_speeds(mean_angle, amp_smooth=31, speed_smooth=51):
    """
    Function to compute the whisking amplitude, speed and whether the animal is pro- or retracting from the extracted whisking envelope.

    Parameters
    ----------
    mean_angle : list
        y coordinates to show global motion of whiskerpad
    amp_smooth : int, default 31 (must be an uequal number)
        defines the size of the window used to smooth the amplitude data.
    speed_smooth : int, default 51 (must be an uequal number)
        defines the size of the window used to smooth the speed data.
    
    Returns
    -------
    all_amplitudes : list
        Whisking position expressed as the amplitude at each particular timepoint. So if the animal is just starting to pro/retract, 
        the amplitude will be low and by the end of the pro/retraction, the amplitude will be higher. I like to use this to plot the whisker movements by position.
    max_ampliudes : list
        Whisking position expressed as the maximum amplitude at each particular pro/retraction. This is useful for getting a measure of overall whisking amplitude.
    whisk_speeds : list
        Whisking speed at each particular timepoint. Calculated by dividing the whisking amplitude by the time it takes to complete it.
    pro_or_ret : list
        Whether the animal is protracting or retracting its whiskers at each particular timepoint

    """

    whisk_params = [[], [], [], []]

    slices, values = misc_tools.detect_contiguity(mean_angle, np.nan)

    for i, slic in enumerate(slices):

        w_params = [[], [], [], []]
        
        if values[i] == 1 and len(mean_angle[slic[0]:slic[1]]) > 50:
            
            pvs, pv_index = whisk_tools.get_prot_and_ret_points(mean_angle[slic[0]:slic[1]])
                    
            intervals = np.diff(pvs, 1) 

            w_w_params = [[np.nan]* pvs[0], [np.nan]* pvs[0], [np.nan]* pvs[0], [np.nan]* pvs[0]]

            for j, (idx, pv) in enumerate(zip(pv_index, pvs)):

                if j < len(intervals):
                    next_pv_dist = intervals[j]
                else:
                    next_pv_dist = slic[1] - (slic[0]+pvs[-1])

                pv_seg_times = np.arange(pv, pv+next_pv_dist) 
                
                if idx == 0:
                    # Protractions
                    each_amp_in_seg = [mean_angle[slic[0]:slic[1]][pv]-mean_angle[slic[0]:slic[1]][pv_seg] for pv_seg in pv_seg_times]
                    max_amps_in_seg = [max(each_amp_in_seg)]*len(each_amp_in_seg)
                    each_speed_in_seg = [each_amp_in_seg[0]] + np.diff(each_amp_in_seg, 1)
                    each_speed_in_seg = np.append(each_speed_in_seg, np.nan)
                    p_or_r = 'PRO'
                if idx == 1:
                    # Retractions
                    each_amp_in_seg = [mean_angle[slic[0]:slic[1]][pv_seg]-mean_angle[slic[0]:slic[1]][pv] for pv_seg in pv_seg_times] 
                    max_amps_in_seg = [max(each_amp_in_seg)]*len(each_amp_in_seg)
                    each_speed_in_seg = [each_amp_in_seg[0]] + np.diff(each_amp_in_seg, 1)
                    each_speed_in_seg = np.append(each_speed_in_seg, np.nan)
                    p_or_r = 'RE'

                w_w_params[0].extend(each_amp_in_seg)
                w_w_params[1].extend(max_amps_in_seg)
                w_w_params[2].extend(each_speed_in_seg)
                w_w_params[3].extend([p_or_r]*len(each_amp_in_seg))

            w_speeds = pd.Series(w_w_params[2]) 
            w_speeds = w_speeds.interpolate(method='linear', limit_area='inside')
            w_w_params[2] = w_speeds.values

            for wp in range(len(w_params)):
                w_params[wp].extend(w_w_params[wp])
        
        else:
            for wp in range(len(w_params)):
                w_params[wp].extend([np.nan]* len(mean_angle[slic[0]:slic[1]]))
        
        for wp in range(len(whisk_params)):
            whisk_params[wp].extend(w_params[wp])
    
    all_max_amps = misc_tools.savgol_smooth(whisk_params[1], window_size=amp_smooth)
    all_whisk_speeds = misc_tools.savgol_smooth(whisk_params[2], window_size=speed_smooth)

    return whisk_params[0], all_max_amps, all_whisk_speeds, whisk_params[3]


def compute_trial_phases_and_freqs(mean_angle, freq_smooth=51):
    """
    Function to compute the whisking phase and frequency from the extracted whisking envelope.

    Parameters
    ----------
    mean_angle : list
        y coordinates to show global motion of whiskerpad
    freq_smooth : int, default 31 (must be an uequal number)
        defines the size of the window used to smooth the frequency data.
    
    Returns
    -------
    all_phases : list
        Whisking phase at each particular timepoint. Maximum Protraction is 1, Maximum Retraction is 0. 
        Values between 0 and 1 mean the animal is protracting. Values between 1 and 2 mean the animal is retracting.
    all_freqs : list
        Whisking position expressed as the maximum amplitude at each particular pro/retraction. This is useful for getting a measure of overall whisking amplitude.

    """

    all_phases = []
    all_freqs = []

    slices, values = misc_tools.detect_contiguity(mean_angle, np.nan)

    for i, slic in enumerate(slices):
        
        if values[i] == 1 and len(mean_angle[slic[0]:slic[1]]) > 50:
            
            pvs, pv_index = whisk_tools.get_prot_and_ret_points(mean_angle[slic[0]:slic[1]])
            
            if len(pvs) > 2:
                phases = whisk_tools.get_phases(pvs, pv_index, mean_angle[slic[0]:slic[1]])       
                whisking_freq = whisk_tools.get_whisking_frequencies(phases, pvs)
            else:
                phases = [np.nan]*len(mean_angle[slic[0]:slic[1]])
                whisking_freq = [np.nan]*len(mean_angle[slic[0]:slic[1]])
        else: 
            phases = [np.nan]* len(mean_angle[slic[0]:slic[1]])
            whisking_freq = [np.nan]* len(mean_angle[slic[0]:slic[1]])

        all_phases.extend(phases)
        all_freqs.extend(whisking_freq)

    all_freqs = misc_tools.savgol_smooth(all_freqs, window_size=freq_smooth)

    return all_phases, all_freqs