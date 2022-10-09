"""
Created on Wed Mar  2 12:01:42 2022

@author: Max
"""

from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

import misc_tools


def collect_peaks(whisk_arr, prominence=0.03, cutoff=4, pixel_scale=100):
    """
    Collect all peaks by their y coordinate and pixel values

    Parameters
    ----------
    whisk_arr : 150 x 500 numpy array
        raw whisker pixel data
    prominence : float, default 0.03
        prominence parameter for scipy.signal.find_peaks
    cutoff : int, default 4
        depending of crop_size, we may want to ignore peaks on the edge of the image.
        From my experience it's a good idea to ignore peaks within 4 pixels of the edge. 
    pixel_scale : float, default 100
        defines by how much pixel values will be scaled to weight their importance in comparison to y-position when tracking peaks.

    Returns
    -------
    peak_coords : list of lists
        each list contains tuples containing y pos and pixel val for peaks at the index of that list 
    raw_img : 150 x 500 numpy array
        raw whisker pixel data

    """
    cutoff = np.shape(whisk_arr)[1] - cutoff
    
    peak_coords = []
    raw_img = []

    for i, line in enumerate(whisk_arr.T):
        if not np.isnan(line).all():
            peaks, _ = signal.find_peaks(-line, prominence=prominence)
            peaks = [p for p in peaks if p <cutoff]
            peak_vals = line[peaks]
    
            peak_coords.append([(p, val*pixel_scale) for p, val in zip(peaks, peak_vals)])
        else:
            peak_coords.append([])
        
        raw_img.append(line)
        
    return peak_coords, raw_img


def peak_thresh(peak_coords):
    """
    Determine a maximum distance threshold for tracking peaks.
    Distances are computed using misc_tools.euclid2 on the tuples of y pos and pixel val.

    Parameters
    ----------
    peak_coords : list of lists
        each list contains tuples containing y pos and pixel val for peaks at the index of that list 

    Returns
    -------
    thresh : float 
        value which is our maximum distance threshold for tracking peaks

    """
    
    peak_dists = []

    for i, peaks in enumerate(peak_coords[:-1]):
        next_peaks = peak_coords[i+1]
        if len(peaks) > 0 and len(next_peaks) > 0:
            for p in peaks:
    
                nearest = min(next_peaks, key=lambda c: (c[0]- p[0])**2 + (c[1]-p[1])**2)
                dist = misc_tools.euclid2(p, nearest)
                peak_dists.append(dist)

    stats = pd.Series(peak_dists).describe()
    upper_thresh = stats['75%'] + (stats['75%'] - stats['25%'])*1.5
    thresh = min([dist for dist in peak_dists if dist > upper_thresh], default = upper_thresh)
    
    return thresh


def track_mmp_slices(peak_coords, thresh, slice_size=3, plot=False):
    """
    Tracks the global motion of peaks using a sliding window technique

    Parameters
    ----------
    peak_coords : list of lists
        each list contains tuples containing y pos and pixel val for peaks at the index of that list 
    thresh : float
        maximum distance threshold for tracking peaks determined via peak_thresh()
    bin_size : int, default 3
        defines the size of the bins used to compute the mean movement of peaks over time sliding windows.
    plot : bool, default False
        If True, plots the output of this function

    Returns
    -------
    mmp_slices : list of lists
        Each list contains a number of tuples of xy coordinates. The number of tuples in each list is equal to the bin_size.
        This list describes the motion of an average tracked peak in that time bin.
        x coordinates are the time (horizontal) coordinates of that average peak.
        y coordinates are the vertical coordinates for the average peak.

    """
    
    mmp_slices = []
    
    for i in range(len(peak_coords[:-slice_size])):
        moving_peaks = []
        
        for j, peaks in enumerate(peak_coords[i:i+slice_size]):
            
            next_peaks = peak_coords[i+j+1]
            if len(peaks) > 0 and len(next_peaks) > 0:
            
                if j == 0:
                    for p in peaks: 
        
                        nearest = min(next_peaks, key=lambda c: (c[0]- p[0])**2 + (c[1]-p[1])**2)
                        dist = misc_tools.euclid2(p, nearest)
        
                        if dist < thresh:
                            moving_peaks.append([p, nearest])
                        else:
                            moving_peaks.append([p, (np.nan, np.nan)])
                else:
                    for mp in moving_peaks:
                        
                        nearest = min(next_peaks, key=lambda c: (c[0]- mp[-1][0])**2 + (c[1]-mp[-1][1])**2)
                        dist = misc_tools.euclid2(mp[-1], nearest)
        
                        if dist < thresh:
                            mp.append(nearest)
                        else:
                            mp.append((np.nan, np.nan))
            
        pslices = []
        if len(moving_peaks) > 2:
            for pslice in moving_peaks:
                if not np.isnan(pslice).any():
        
                    ps_list = np.array([c[0] for c in pslice])
                    pslices.append(ps_list)
            mean_slice = np.mean(pslices, axis=0)
            
            if isinstance(mean_slice, np.float64):
                mean_peak_coords = ([])
            else:
                times = np.linspace(i, i+slice_size, slice_size+1)
    
                #Handling end cases
                idx_diff = len(times) - len(mean_slice)
                if idx_diff > 0:
                    times = times[:-idx_diff]
                
                
                mean_peak_coords = [(t, ms) for t, ms in zip(times,mean_slice)]
                if plot:
                    plt.plot(times, mean_slice)
        
        else:
            mean_peak_coords = ([])
            
        mmp_slices.append(mean_peak_coords)
        
    if plot:
        plt.gca().invert_yaxis()
        plt.show()
        
    return mmp_slices


def mean_peakpos(mmp_slices, limits=(0,30)):
    """
    determine the mean peak y position based on all the mean tracked peak motion in a section of the trial.
    Not the ideal way to do things...

    Parameters
    ----------
    mmp_slices : list of lists
        Each list contains a number of tuples of xy coordinates. The number of tuples in each list is equal to the bin_size.
        This list describes the motion of an average tracked peak in that time bin.
        x coordinates are the time (horizontal) coordinates of that average peak.
        y coordinates are the vertical coordinates for tracked peaks.
    limits : tuple, default (0,30)
        Upper and lower limit for section of trial where we want to average from.

    Returns
    -------
    mean : float
        mean value

    """
    
    firstNonEmpty = misc_tools.first_non_empty(mmp_slices)

    return np.mean([np.mean([x[1] for x in slic]) for slic in mmp_slices[firstNonEmpty+limits[0]:
                                                                         firstNonEmpty+limits[1]]])
        

def patch_mmp_slices(mmp_slices, start_pos, raw_img, min_bout_length = 20, plot = False):
    """
    Patch up the mean moving peak traces into a coherent line

    Parameters
    ----------
    mmp_slices : list of lists
        Each list contains a number of tuples of xy coordinates. The number of tuples in each list is equal to the bin_size.
        This list describes the motion of an average tracked peak in that time bin.
        x coordinates are the time (horizontal) coordinates of that average peak.
        y coordinates are the vertical coordinates for tracked peaks.
    start_pos : float
        defines the starting point for the whisking envelope.
        Currently working on a way to find the true absolute whisking setpoint.
    raw_img : 150 x 500 numpy array
        raw whisker pixel data
    min_bout_length : int, default 200
        Only keep patched up bouts that have a length of min_bout_length or more.
    plot : bool, default False
        if True, plots the output of this function over the raw data

    Returns
    -------
    patched_mmp_slices : list of lists
        Each list contains a number of tuples of xy coordinates. The number of tuples in each list is equal to the bin_size.
        This list describes the motion of an average tracked peak in that time bin.
        x coordinates are the time (horizontal) coordinates of that average peak.
        y coordinates are the vertical coordinates for tracked peaks.
        y coordinates have also been 'patched up' giving the appearance of a continuous line
    """
    
    if plot:
        plt.figure()
        plt.imshow(np.array(raw_img).T)
    
    contiguity_line = []
    for l in mmp_slices:
        if len(l) == 0:
            contiguity_line.append(np.nan)
        else:
            contiguity_line.append(1)
    slices, values = misc_tools.detect_contiguity(contiguity_line, np.nan)
    valid_slices = [slices for i, slices in enumerate(slices) if values[i]==1 and slices[1]-slices[0] > min_bout_length]
    
    patched_mmp_slices = []
    for vslice in valid_slices:
    
        for i, slic in enumerate(mmp_slices[vslice[0]:vslice[1]-1]):

            xs = [x[0] for x in slic]
            
            if i == 0:
                first_y = [x[1] for x in slic][0]
                first_diff = start_pos-first_y
                ys = [x[1]+first_diff for x in slic]  
                patched_next_ys = ys
            else:
                ys =  patched_next_ys
            
            patched_mmp_slices.append([(x, y) for x, y in zip(xs, ys)])
                
            y = ys[1]
            
            if plot:
                plt.plot(xs,ys)
            
            next_ys = [x[1] for x in mmp_slices[i+vslice[0]+1]]
            
            next_y = next_ys[0]
            
            diff = next_y-y
            
            patched_next_ys = next_ys - diff
    
    if plot:
        plt.show()
    
    return patched_mmp_slices


def average_mmp_slices(patched_mmp_slices, x_range):
    """
    Average our patched up mean moving peak traces into a single line plot

    Parameters
    ----------
    patched_mmp_slices : list of lists
        Each list contains a number of tuples of xy coordinates. The number of tuples in each list is equal to the bin_size.
        This list describes the motion of an average tracked peak in that time bin.
        x coordinates are the time (horizontal) coordinates of that average peak.
        y coordinates are the vertical coordinates for tracked peaks.
        y coordinates have also been 'patched up' giving the appearance of a continuous line
    x_range : tuple
        defines the range of x values to average over for

    Returns
    -------
    mean_angle : list
        y coordinates to show global motion of whiskerpad

    """
    
    mean_angle = []
    for i in range(x_range[0], x_range[1]):
        vals_at_i = []
        for slic in patched_mmp_slices:
            vals_at_i.extend([x[1] for x in slic if x[0] == i])

        mean_angle.append(np.nanmean(vals_at_i))  
    
    return mean_angle


def find_peaks_and_valleys(data):
    """
    Find peaks and valleys in a list of data
    
    Parameters
    ----------
    data : list
        list of values

    Returns
    -------
    peaks : list
        list of peak indices
    valleys : list
        list of valley indices

    """
    
    peaks, _ = signal.find_peaks(data)
    valleys, _ = signal.find_peaks(-data)
    
    return peaks, valleys


def return_pvs(peaks, valleys):
    """
    Create two lists, one to index where peaks and valleys are, the other to index whether it's a peak or a valley
    
    Parameters
    ----------
    peaks : list
        list of peak indices
    valleys : list
        list of valley indices

    Returns
    -------
    pvs : list
        list of indices where peaks and valleys are
    pv_index : list
        list of 1s and 0s to show whether it's a peak or a valley

    """

    pvs = sorted(list(peaks) + list(valleys))

    ## combine and sort lists
    pv_index = []
    for pv in pvs:
        if pv in peaks:
            pv_index.append(1)
        if pv in valleys:
            pv_index.append(0)
            
    return pvs, pv_index


def remove_invalid_pvs(pvs, pv_index, mean_angle):
    """
    Remove 'invalid' peaks and valleys from our data. Function makes sure we still have alternating peaks and valleys.
    
    Parameters
    ----------
    pvs : list
        list of indices where peaks and valleys are
    pv_index : list
        list of 1s and 0s to show whether it's a peak or a valley
    mean_angle : list
        y coordinates to show global motion of whiskerpad

    Returns
    -------
    new_pvs : list
        list of indices where peaks and valleys are
    new_pv_index : list
        list of 1s and 0s to show whether it's a peak or a valley

    """
    
    intervals = np.diff(pvs, 1) 
    slices, values = misc_tools.detect_contiguity(intervals, 2)
    slices_to_clean = [slices for i, slices in enumerate(slices) if values[i]==0]

    pvs_to_remove = []
    for slic in slices_to_clean:
        slic_check = pvs[slic[0]:slic[1]+1]
        idx_slic_check = pv_index[slic[0]:slic[1]+1]

        peaks = [x for x, y in zip(slic_check, idx_slic_check) if y == 1]
        valleys = [x for x, y in zip(slic_check, idx_slic_check) if y == 0]
        
        if len(slic_check) > 2:
            if len(peaks) > len(valleys):
                idx_to_keep = np.argmin(np.array(mean_angle)[peaks])
                slic_check.pop(slic_check.index(peaks[idx_to_keep]))
            if len(peaks) < len(valleys):
                idx_to_keep = np.argmax(np.array(mean_angle)[valleys])
                slic_check.pop(slic_check.index(valleys[idx_to_keep]))
            
        pvs_to_remove.extend(slic_check)
    
    new_pvs = [pv for pv in pvs if pv not in pvs_to_remove]
    new_pv_index = [i for i, pv in zip(pv_index, pvs) if pv not in pvs_to_remove]
    
    return new_pvs, new_pv_index


def get_prot_and_ret_points(mean_angle):
    """
    Get protraction and retraction points from our whisking envelope data
    
    Parameters
    ----------
    mean_angle : list
        y coordinates to show global motion of whiskerpad

    Returns
    -------
    pvs : list
        list of indices where peaks and valleys are
    pv_index : list
        list of 1s and 0s to show whether it's a peak or a valley

    """
    
    peaks, valleys = find_peaks_and_valleys(-np.array(mean_angle))
    pvs, pv_index = return_pvs(peaks, valleys)

    if len(pvs) > 0:
        pvs, pv_index = remove_invalid_pvs(pvs, pv_index, mean_angle)
        
    return pvs, pv_index


#def get_prots_and_rets(mean_angle, pvs):
#    
#    next_pvs = pvs[1:] + [np.nan]
#    pv_amplitudes =[mean_angle[pv]-mean_angle[n_pv] for pv, n_pv in zip(pvs[:-1], next_pvs[:-1])]
#    pv_times = np.diff(pvs, 1) 
#    
#    return pv_amplitudes, pv_times


#def split_prots_and_rets(pv_amplitudes, pv_times):
#    
#    prot_amps = [i for i in pv_amplitudes if i > 0]
#    ret_amps = [i for i in pv_amplitudes if i < 0]
#    prot_times = [j for i, j in zip(pv_amplitudes, pv_times) if i > 0]
#    ret_times = [j for i, j in zip(pv_amplitudes, pv_times) if i < 0]
#    
#    return prot_amps, prot_times, ret_amps, ret_times


def get_phases(pvs, pv_index, mean_angle):
    """
    Compute whisking phases from our whisking envelope data.

    Parameters
    ----------
    pvs : list
        list of indices where peaks and valleys are
    pv_index : list
        list of 1s and 0s to show whether it's a peak or a valley
    mean_angle : list
        y coordinates to show global motion of whiskerpad

    Returns
    -------
    value_array : list
        Describes the whisking phase at each particular point in time.
        Maximum Protraction is 1, Maximum Retraction is 0. 
        Values between 0 and 1 mean the animal is protracting. 
        Values between 1 and 2 mean the animal is retracting. 

    """
    
    indexes = np.arange(pvs[0], pvs[-1])

    phases = []
    for i in indexes:
        if i in pvs:
            if pv_index[pvs.index(i)] == 0:
                phases.append(0)
            if pv_index[pvs.index(i)] == 1:
                phases.append(1)
        else:
            phases.append(np.nan)


    for idx, (i, p) in enumerate(zip(indexes, phases)):

        if not np.isnan(p):
            start_phase = mean_angle[i]
            end_phase = mean_angle[pvs[pvs.index(i)+1]]
            if p == 0:
                interval = start_phase - end_phase
                prore = 'PRO'
            if p == 1:
                interval = end_phase - start_phase
                prore = 'RE'

        if np.isnan(p):
            relative_mean_angle = (mean_angle[i] - min(start_phase, end_phase))/interval
            if prore == 'PRO':
                relative_mean_angle = 1-relative_mean_angle
            if prore == 'RE':
                relative_mean_angle = relative_mean_angle+1

            phases[idx] = relative_mean_angle
            
    value_array = np.empty(len(mean_angle))
    value_array[:] = np.nan
    
    value_array[pvs[0]:pvs[-1]] = phases
            
    return value_array


def get_whisking_frequencies(phases, pvs):
    """
    Compute whisking phases from our whisking envelope data.

    Parameters
    ----------
    value_array : list
        Describes the whisking phase at each particular point in time.
        Maximum Protraction is 1, Maximum Retraction is 0. 
        Values between 0 and 1 mean the animal is protracting. 
        Values between 1 and 2 mean the animal is retracting. 
    pvs : list
        list of indices where max protraction and retraction points are

    Returns
    -------
    whisking_freq : list
        Whisking frequency at each particular timepoint in Hz (Assuming sampling frequency was 500Hz).

    """

    whisking_freq = np.empty(len(phases))
    whisking_freq[:] = np.nan
    phases_corrected = phases.copy()
    phases_corrected[phases_corrected == 0] = 2

    if len(pvs) > 2:

        for i, ph in enumerate(phases[pvs[1]:pvs[-3]]):
            current_idx = i+pvs[1]
            if not np.isnan(ph):

                if ph == 1:

                    half_phase_shift = ph - 1
                    next_idx = misc_tools.first_index_equal(phases[current_idx:], half_phase_shift)

                    next_idx = current_idx+next_idx

                    prev_idx = misc_tools.first_index_equal(phases[:current_idx][::-1], half_phase_shift)
                    prev_idx = current_idx-prev_idx-1

                if ph > 1:

                    half_phase_shift = ph - 1
                    next_pv = misc_tools.first_index_above(pvs, current_idx, 500)

                    next_idx = misc_tools.first_index_above(phases[current_idx:pvs[next_pv+1]], half_phase_shift, 1)
                    if next_idx is None:
                        next_idx = pvs[next_pv+1]
                    else:
                        next_idx = current_idx+next_idx

                    m = misc_tools.slope(next_idx-1, phases[next_idx-1], next_idx, phases[next_idx])
                    b = misc_tools.intercept(next_idx-1, phases[next_idx-1], m)

                    next_idx = (half_phase_shift - b)/m

                    prev_idx = misc_tools.first_index_under(phases[pvs[next_pv-2]:current_idx][::-1], half_phase_shift, 0)
                    if prev_idx is None:
                        prev_idx = pvs[next_pv-2]
                    else:
                        prev_idx = current_idx-prev_idx-1

                    m = misc_tools.slope(prev_idx, phases[prev_idx], prev_idx+1, phases[prev_idx+1])
                    b = misc_tools.intercept(prev_idx, phases[prev_idx], m)

                    prev_idx = (half_phase_shift - b)/m

                if ph == 0:

                    half_phase_shift = ph + 1
                    next_idx = misc_tools.first_index_equal(phases[current_idx:], half_phase_shift)
                    next_idx = current_idx+next_idx

                    prev_idx = misc_tools.first_index_equal(phases[:current_idx][::-1], half_phase_shift)
                    prev_idx = current_idx-prev_idx-1
                    

                if ph > 0 and ph < 1:

                    half_phase_shift = ph + 1
                    next_pv = misc_tools.first_index_above(pvs, current_idx, 500)

                    next_idx = misc_tools.first_index_above(phases[current_idx:pvs[next_pv+1]], half_phase_shift, 2)
                    if next_idx is None:
                        next_idx = pvs[next_pv+1]
                    else:
                        next_idx = current_idx+next_idx

                    m = misc_tools.slope(next_idx-1, phases_corrected[next_idx-1], next_idx, phases_corrected[next_idx])
                    b = misc_tools.intercept(next_idx-1, phases_corrected[next_idx-1], m)

                    next_idx = (half_phase_shift - b)/m    

                    prev_idx = misc_tools.first_index_under(phases[pvs[next_pv-2]:current_idx][::-1], half_phase_shift, 1)
                    if prev_idx is None:
                        prev_idx = pvs[next_pv-2]
                    else:
                        prev_idx = current_idx-prev_idx-1

                    m = misc_tools.slope(prev_idx, phases_corrected[prev_idx], prev_idx+1, phases_corrected[prev_idx+1])
                    b = misc_tools.intercept(prev_idx, phases_corrected[prev_idx], m)

                    prev_idx = (half_phase_shift - b)/m  
            
                freq = 500 / (next_idx - prev_idx)

            whisking_freq[current_idx] = freq
    
    return whisking_freq
        