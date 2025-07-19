import sys, os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from itertools import product
from scipy.optimize import linear_sum_assignment
import scipy.signal as signal
import pandas as pd
import whisk_params, whisk_tools, misc_tools, whisk_setpoint
from scipy.interpolate import interp1d
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
import copy # Import copy for deep copies of Track objects

# Add your packages path
sys.path.append(r'C:\Users\Max\Documents\Python\__packages__')

# Import your libraries
import mLabAna as mana

# Load session data
SD = mana.SessionData(4204)
print(SD.mouse)

count = 0

all_nrs_of_tracelets = []

# Process each trial
for index, row in SD.iterrows():
    print(index)
    if type(row['WhiskerTracking_Results']) == np.ndarray:
        whisker_results = row['WhiskerTracking_Results']

        #plt.imshow(whisker_results)
        #plt.show()

        whisker_movement_envelope = whisk_params.extract_whisking_envelope(whisker_results, prominence=0.1, cutoff=4, pixel_scale=0.5)
        #plt.imshow(whisker_results)
        #plt.plot(whisker_movement_envelope, c='r')
        #plt.show()

        whisker_setpoint_smoothed, whisker_spread, nr_of_tracelets = whisk_setpoint.extract_setpoint_and_spread(whisker_results, 
                                                                                                                whisker_movement_envelope, 
                                                                                                                prominence=5, 
                                                                                                                cutoff=0, 
                                                                                                                pixel_scale=0.5, 
                                                                                                                max_distance=4, 
                                                                                                                max_inactive=3, 
                                                                                                                min_length=10, 
                                                                                                                history_window=6, 
                                                                                                                history_threshold=0.5)
                                                            
        #plt.imshow(whisker_results)
        #plt.plot(whisker_setpoint_smoothed, c='r')
        #plt.show()

        #plt.plot(whisker_spread)
        #plt.show()

        # Compute whisking amplitudes and speeds
        whisk_amps, max_amps, whisk_speeds, pro_or_ret = whisk_params.compute_trial_amps_and_speeds(whisker_setpoint_smoothed, speed_smooth=51)                

        #plt.plot(whisk_amps)
        #plt.show()

        #plt.plot(max_amps)
        #plt.show()

        #plt.plot(whisk_speeds)
        #plt.show()

        # Compute whisking phases and frequencies
        whisk_phases, inst_freqs = whisk_params.compute_trial_phases_and_freqs(whisker_setpoint_smoothed, freq_smooth=51)


        #plt.plot(whisk_phases)
        #plt.show()

        #plt.plot(inst_freqs)
        #plt.show()
        
        #count += 1
        #if count > 50:
        #   break

#most_common_value = max(set(all_nrs_of_tracelets), key=all_nrs_of_tracelets.count)
#print(f"Most common number of tracelets: {most_common_value}")

# New code to stitch up tracelets for each trial based on most common value


