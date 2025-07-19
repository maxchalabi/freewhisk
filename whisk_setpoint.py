import matplotlib.pyplot as plt
import numpy as np
import whisk_tools, misc_tools
from scipy.interpolate import interp1d
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
import copy # Import copy for deep copies of Track objects
import pandas as pd



# Main function to extract the whsiking setpoint tracelets
def extract_setpoint_and_spread(whisk_arr, mean_angle, prominence=5, cutoff=0, pixel_scale=0.5, max_distance=4, max_inactive=3, min_length=10, history_window=6, history_threshold=0.5):
    """
    Extract the whisking setpoint from raw whisker pixel data using the global envelope as a template.

    Parameters
    ----------
    whisk_arr : 150 x 500 numpy array
        Raw whisker pixel data.
    prominence : float, default 0.03
        Prominence parameter for peak detection.
    cutoff : int, default 4
        Ignore peaks near the edge.
    pixel_scale : float, default 0.5
        Scaling factor for pixel values in distance metric.
    max_distance : float, default 10
        Maximum distance for peak association.
    max_inactive : int, default 5
        Maximum number of time points a track can be inactive before termination.
    min_length : int, default 50
        Minimum length of a valid track.
    history_window : int, default 8
        Number of recent frames to check for inactivity pattern.
    history_threshold : float, default 0.5
        Threshold for the ratio of missed frames in the history window.

    Returns
    -------
    individual_traces : list of lists
        Each sublist contains y-positions of an individual whisker trace over time.
    """
    
    # Collect peaks
    peak_coords, _ = whisk_tools.collect_peaks(whisk_arr, prominence, cutoff, pixel_scale)
    #plt.imshow(whisk_arr)
    #for t in range(len(peak_coords)):
    #    if len(peak_coords[t]) > 0:
    #        for peak in peak_coords[t]:
    #            plt.scatter(t, peak[0], c='r', s=1)
    #plt.show()
    
    normalized_peaks = normalize_all_peaks(peak_coords, mean_angle)

    # Plot the normalized peaks over time
    #fig, ax = plot_normalized_peaks(normalized_peaks)
    #plt.show()

    completed_tracelets = track_peaks(normalized_peaks, max_inactive=max_inactive, max_distance=max_distance, 
                                    min_length=min_length, history_window=history_window, 
                                    history_threshold=history_threshold)

    #fig, ax = plot_normalized_traces(normalized_peaks, completed_tracelets)
    #plt.show()

    #sort completed traces by length
    completed_tracelets = sorted(completed_tracelets, key=lambda x: len(x.positions), reverse=True)

    denormalized_tracelets = denormalize_tracelets(completed_tracelets, mean_angle)

    #fig, ax = plot_denormalized_tracelets(whisk_arr, denormalized_tracelets)
    #plt.show()

    # Sort traces by length
    denormalized_tracelets = sorted(denormalized_tracelets, key=lambda x: len(x.positions), reverse=True)
    
    
    #fig, ax = plot_denormalized_tracelets(whisk_arr, denormalized_tracelets[:1])
    #plt.show()
    
    separated_tracelets = get_separated_tracelets(denormalized_tracelets)

    #fig, ax = plot_denormalized_tracelets(whisk_arr, separated_tracelets)
    #plt.show()

    slices, values = misc_tools.detect_contiguity(mean_angle, np.nan)
    valid_slices = [slice for slice, value in zip(slices, values) if not(np.isnan(value))]

    #plt.imshow(whisk_arr, cmap='gray')

    nr_of_tracelets = np.zeros(len(mean_angle))
    nr_of_tracelets[:] = np.nan

    all_stitched_traces = []
    all_stitched_upper_traces = []
    all_stitched_lower_traces = []
    all_stitched_ranges = []
    all_stitched_upper_ranges = []
    all_stitched_lower_ranges = []
    for vslice in valid_slices:
        window_traces, window_ranges, nr_of_samples_in_slice = average_window_traces(separated_tracelets, vslice)
        upper_window_traces, lower_window_traces, all_separated_window_ranges, _ = average_lower_upper_window_traces(separated_tracelets, vslice)
        
        # Plot each window trace directly
        #for window_range, window_trace in zip(window_ranges, window_traces):
        #    plt.plot(window_range, window_trace, c='red')

        #for window_range, upper_window_trace, lower_window_trace in zip(all_separated_window_ranges, upper_window_traces, lower_window_traces):
        #    plt.plot(window_range, upper_window_trace, c='green')
        #    plt.plot(window_range, lower_window_trace, c='blue')

        if valid_slices:
            nr_of_tracelets[vslice[0]:vslice[1]-2] = nr_of_samples_in_slice

        stitched_range, stitched_trace = stitch_traces(window_ranges, window_traces)        
        stitched_upper_range, stitched_upper_trace = stitch_traces(all_separated_window_ranges, upper_window_traces)
        stitched_lower_range, stitched_lower_trace = stitch_traces(all_separated_window_ranges, lower_window_traces)
        all_stitched_traces.append(stitched_trace)
        all_stitched_upper_traces.append(stitched_upper_trace)
        all_stitched_lower_traces.append(stitched_lower_trace)
        all_stitched_ranges.append(stitched_range)
        all_stitched_upper_ranges.append(stitched_upper_range)
        all_stitched_lower_ranges.append(stitched_lower_range)

    #plt.show()

    #plt.plot(nr_of_tracelets, c='r')
    #plt.axhline(y=np.median(nr_of_samples_in_slice), c='b', linestyle='--')
    #plt.show()


    #plt.imshow(whisk_arr, cmap='gray')
    #for stitched_range, stitched_trace in zip(all_stitched_ranges, all_stitched_traces):
    #    plt.plot(stitched_range, stitched_trace, c='r')
    #plt.show()


    whisker_setpoint_iter1 = np.zeros(len(mean_angle))
    whisker_setpoint_iter1[:] = np.nan
    for stitched_range, stitched_trace in zip(all_stitched_ranges, all_stitched_traces):
        whisker_setpoint_iter1[stitched_range] = stitched_trace
    whisker_setpoint_smoothed = misc_tools.savgol_smooth(whisker_setpoint_iter1, window_size=11)
    #plt.imshow(whisk_arr, cmap='gray')
    #plt.plot(whisker_setpoint_smoothed, c='r')
    #plt.show()

    #plt.imshow(whisk_arr, cmap='gray')
    #for stitched_range, stitched_upper_trace, stitched_lower_trace in zip(all_stitched_upper_ranges, all_stitched_upper_traces, all_stitched_lower_traces):
    #    plt.plot(stitched_range, stitched_upper_trace, c='g')
    #    plt.plot(stitched_range, stitched_lower_trace, c='b')
    #plt.show()

    whisker_upper_setpoint = np.zeros(len(mean_angle))
    whisker_upper_setpoint[:] = np.nan
    whisker_lower_setpoint = np.zeros(len(mean_angle))
    whisker_lower_setpoint[:] = np.nan
    for stitched_range, stitched_upper_trace, stitched_lower_trace in zip(all_stitched_upper_ranges, all_stitched_upper_traces, all_stitched_lower_traces):
        whisker_upper_setpoint[stitched_range] = stitched_upper_trace
        whisker_lower_setpoint[stitched_range] = stitched_lower_trace
    whisker_upper_setpoint_smoothed = misc_tools.savgol_smooth(whisker_upper_setpoint, window_size=11)
    whisker_lower_setpoint_smoothed = misc_tools.savgol_smooth(whisker_lower_setpoint, window_size=11)

    #plt.imshow(whisk_arr, cmap='gray')
    #plt.plot(whisker_upper_setpoint_smoothed, c='g')
    #plt.plot(whisker_lower_setpoint_smoothed, c='b')
    #plt.show()

    whisker_spread = whisker_lower_setpoint_smoothed-whisker_upper_setpoint_smoothed
    #plt.plot(whisker_spread, c='r')
    #plt.show()

    return whisker_setpoint_smoothed, whisker_spread, nr_of_tracelets



# New Track class for managing individual whisker traces
class Track:
    def __init__(self, initial_position, time):
        self.positions = [initial_position]
        self.times = [time]
        self.active = True
        self.inactive_count = 0
        self.update_history = [True]  # Track whether each frame was updated (True) or missed (False)

    def predict(self, time=None):
        return self.positions[-1]

    def update(self, new_position, time):
        self.positions.append(new_position)
        self.times.append(time)
        self.inactive_count = 0
        self.update_history.append(True)

    def mark_inactive(self, max_inactive=5, history_window=8, history_threshold=0.5):
        self.inactive_count += 1
        self.update_history.append(False)
        
        # Original condition: too many consecutive missed frames
        if self.inactive_count > max_inactive:
            self.active = False
            return
            
        # New condition: too many missed frames in recent history
        if len(self.update_history) >= history_window:
            recent_history = self.update_history[-history_window:]
            missed_ratio = 1 - (sum(recent_history) / len(recent_history))
            if missed_ratio > history_threshold:
                self.active = False
                
    def duplicate(self):
        """
        Create a copy of this track with the same data
        """
        # Create a new track with the first position
        new_track = Track(self.positions[0], self.times[0])
        
        # Directly copy the lists (slicing creates new lists)
        new_track.positions = self.positions.copy()
        new_track.times = self.times.copy()
        new_track.update_history = self.update_history.copy()
        
        # Copy metadata
        new_track.active = self.active
        new_track.inactive_count = self.inactive_count
        
        return new_track
    
    def modify_last_point(self, position=None, time=None):
        """
        Modify the last data point in the track
        
        Parameters
        ----------
        position : float, optional
            New position value for the last point
        time : float, optional
            New time value for the last point
        """
        if len(self.positions) > 0:
            if position is not None:
                self.positions[-1] = position
            if time is not None:
                self.times[-1] = time



# Function to stitch overlapping traces using least squares
def stitch_traces(window_ranges, window_traces):
    """
    Stitches overlapping trace segments by finding optimal vertical offsets
    that minimize discrepancies in overlapping regions using least squares.

    Parameters
    ----------
    window_ranges : list of lists/ranges
        List where each element contains the time indices for a trace segment.
    window_traces : list of numpy arrays
        List where each element is a trace segment (y-values).

    Returns
    -------
    final_range : list
        List of time indices for the stitched trace.
    final_trace : numpy array
        The stitched trace (y-values). Returns empty list/array if input is empty.
        Returns the original trace if only one segment is provided.
        Returns the first trace if no overlaps are found between multiple segments.
    """
    num_traces = len(window_traces)
    if num_traces == 0:
        return [], np.array([])
    if num_traces == 1:
        # Ensure range is a list for consistency
        return list(window_ranges[0]), window_traces[0]

    # Map global time index to relative index within each trace segment
    trace_time_maps = []
    for r, tr in zip(window_ranges, window_traces):
        trace_time_maps.append({t: idx for idx, t in enumerate(r)})

    equations_rows = []
    b_values = []

    # Build equations based on overlaps
    for i in range(num_traces):
        for j in range(i + 1, num_traces):
            range_i = set(window_ranges[i])
            range_j = set(window_ranges[j])
            overlap_times = sorted(list(range_i.intersection(range_j)))

            if len(overlap_times) > 0:
                map_i = trace_time_maps[i]
                map_j = trace_time_maps[j]
                trace_i = window_traces[i]
                trace_j = window_traces[j]

                for t in overlap_times:
                    # Equation: offset_i - offset_j = trace_j(t) - trace_i(t)
                    row = np.zeros(num_traces)
                    row[i] = 1
                    row[j] = -1
                    equations_rows.append(row)

                    idx_i = map_i[t]
                    idx_j = map_j[t]
                    # Check if indices are valid (can happen with extrapolation issues perhaps)
                    if 0 <= idx_i < len(trace_i) and 0 <= idx_j < len(trace_j):
                         b_values.append(trace_j[idx_j] - trace_i[idx_i])
                    else:
                         # If indices are out of bounds, append a row of zeros to A
                         # and 0 to b to effectively ignore this point.
                         equations_rows[-1] = np.zeros(num_traces) # Zero out last added row
                         b_values.append(0)


    if not equations_rows or np.all(np.array(equations_rows) == 0): # No valid overlaps found
         print(f"Warning: No valid overlaps found for stitching {num_traces} traces. Returning the first trace.")
         # Find trace with the most points as a fallback? Or just first? Let's stick to first for now.
         # Return range as list
         return list(window_ranges[0]), window_traces[0]

    # Convert to sparse matrix
    A = lil_matrix((len(equations_rows), num_traces), dtype=float)
    valid_eq_index = 0
    for i, row_data in enumerate(equations_rows):
         # Only add rows that are not all zeros (from invalid index check)
         if np.any(row_data != 0):
             A[valid_eq_index, :] = row_data
             valid_eq_index += 1
         else:
             # Ensure b_values matches the rows kept in A
             b_values.pop(i - (len(equations_rows) - valid_eq_index))

    if valid_eq_index == 0: # Check again after potential filtering
        print(f"Warning: No valid equations after filtering. Returning the first trace.")
        return list(window_ranges[0]), window_traces[0]

    A = A[:valid_eq_index, :].tocsr() # Trim unused rows and convert format
    b = np.array(b_values)


    # Solve the least squares problem: Ax = b for offsets x
    try:
        # Using lsqr suitable for sparse matrices, finds minimum norm solution
        result = lsqr(A, b, damp=1e-3)
        offsets = result[0]
        # Check if the solution converged properly (istop gives info)
        if result[1] > 2: # 1=exact, 2=within tolerance, >2 might indicate issues
             print(f"Warning: lsqr convergence issue (istop={result[1]}). Offsets may be inaccurate.")

    except Exception as e:
        print(f"Error during lsqr solve: {e}. Returning first trace.")
        return list(window_ranges[0]), window_traces[0]

    # Apply offsets
    adjusted_traces = []
    for i in range(num_traces):
        adjusted_traces.append(window_traces[i] + offsets[i])

    # Combine adjusted traces by averaging overlaps
    all_times = sorted(list(set(t for r in window_ranges for t in r)))
    if not all_times:
        return [], np.array([])
        
    min_time = min(all_times)
    max_time = max(all_times)
    final_range = list(range(min_time, max_time + 1))
    final_trace = np.zeros(len(final_range)) # Initialize with zeros
    trace_counts = np.zeros(len(final_range))

    for i in range(num_traces):
        current_range = window_ranges[i]
        current_adjusted_trace = adjusted_traces[i]
        for t_idx, t in enumerate(current_range):
            if min_time <= t <= max_time: # Ensure time is within the final range
                final_idx = t - min_time
                # Check if trace value is valid before adding
                if not np.isnan(current_adjusted_trace[t_idx]):
                     final_trace[final_idx] += current_adjusted_trace[t_idx]
                     trace_counts[final_idx] += 1

    # Average where traces overlapped, handle NaNs
    final_trace = np.divide(final_trace, trace_counts, where=trace_counts > 0, out=np.full_like(final_trace, np.nan))


    slices, values = misc_tools.detect_contiguity(final_trace , np.nan)
    valid_slices = [slice for slice, value in zip(slices, values) if not(np.isnan(value))]
    for vslice in valid_slices:
        if np.abs(vslice[0]-vslice[1]) < 20:
            final_trace[vslice[0]:vslice[1]] = np.nan
    if len(valid_slices) == 2:
        final_trace = interpolate_missing(final_trace)

    return final_range, final_trace



def interpolate_missing(data, method='cubic'):
    """
    Interpolates missing values in a list using the specified method.
    
    Parameters:
    data (list): List of values with missing data represented as NaNs.
    method (str): Interpolation method to use. Default is 'cubic'.
                  Other options include 'linear', 'nearest', 'slinear', 
                  'quadratic', 'barycentric', 'polynomial', etc.
    
    Returns:
    list: Interpolated list with missing values filled.
    
    Notes:
    - Assumes the data corresponds to evenly spaced indices (e.g., time steps).
    - NaNs at the start or end may remain unfilled depending on the method.
    """
    # Convert the input list to a pandas Series
    series = pd.Series(data)
    # Perform interpolation with the specified method
    interpolated = series.interpolate(method=method)
    # Convert back to a list and return
    return interpolated.tolist()



def hierarchical_stitch_traces(window_ranges, window_traces, small_window_size=20, medium_window_size=100, overlap_percent=20, whisker_array=None):
    """
    Stitches overlapping trace segments hierarchically in three iterations:
    1. Small subsegments, 2. Medium subsegments, 3. Final stitched trace.

    Parameters
    ----------
    window_ranges : list of lists/ranges
        List where each element contains the time indices for a trace segment.
    window_traces : list of numpy arrays
        List where each element is a trace segment (y-values).
    small_window_size : int, default 50
        The size of the smallest subsegment window in time units.
    medium_window_size : int, default 150
        The size of the medium subsegment window in time units.
    overlap_percent : float, default 50
        The percentage of overlap between consecutive subsegments at each level.

    Returns
    -------
    final_range : list
        List of time indices for the stitched trace.
    final_trace : numpy array
        The stitched trace (y-values). Returns empty list/array if input is empty.
    """
    num_traces = len(window_traces)
    if num_traces == 0:
        return [], np.array([])
    if num_traces == 1:
        return list(window_ranges[0]), window_traces[0]

    # Determine the total time range
    all_times = sorted(list(set(t for r in window_ranges for t in r)))
    if not all_times:
        return [], np.array([])
    min_time = min(all_times)
    max_time = max(all_times)
    total_range = max_time - min_time + 1

    # If the total range is smaller than the small window size, revert to regular stitching
    if total_range <= small_window_size:
        return stitch_traces(window_ranges, window_traces)

    # Calculate overlap sizes for each level
    small_overlap_size = int(small_window_size * overlap_percent / 100)
    small_step_size = small_window_size - small_overlap_size
    medium_overlap_size = int(medium_window_size * overlap_percent / 100)
    medium_step_size = medium_window_size - medium_overlap_size

    # If the total range is smaller than the medium window size, skip medium level
    use_medium_level = total_range > medium_window_size

    if whisker_array is not None:
        # Create a figure to visualize the subsegments
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
        ax1.set_title('Original Traces and Small Subsegment Windows')
        ax2.set_title('Small Stitched Subsegments and Medium Windows')
        ax3.set_title('Medium Stitched Subsegments and Final Stitch')
        ax1.imshow(whisker_array, cmap='gray')
        ax2.imshow(whisker_array, cmap='gray')
        ax3.imshow(whisker_array, cmap='gray')
        
        # Plot original traces
        for i, (r, t) in enumerate(zip(window_ranges, window_traces)):
            if len(r) > 0:
                ax1.plot(r, t, alpha=0.5, label=f'Trace {i}' if i < 10 else None)

    # First Level: Split into small overlapping windows
    small_windows = []
    start = min_time
    while start < max_time:
        end = min(start + small_window_size - 1, max_time)
        small_windows.append((start, end))
        start += small_step_size

    if whisker_array is not None:
        # Plot small subsegment windows
        for i, (start, end) in enumerate(small_windows):
            ax1.axvspan(start, end, alpha=0.2, color=f'C{i%10}')
            ax1.text(start + (end-start)/2, ax1.get_ylim()[0] + (ax1.get_ylim()[1]-ax1.get_ylim()[0])*0.05, 
                    f'S{i}', ha='center', va='bottom', fontsize=8)

    # Stitch traces within each small subsegment window
    small_subsegment_ranges = []
    small_subsegment_traces = []
    for i, (start, end) in enumerate(small_windows):
        # Filter traces that fall within or overlap with this window
        sub_ranges = []
        sub_traces = []
        for r, t in zip(window_ranges, window_traces):
            r_min = min(r) if r else start
            r_max = max(r) if r else end
            if r_max >= start and r_min <= end:
                sub_ranges.append(r)
                sub_traces.append(t)
        
        if sub_ranges:
            stitched_range, stitched_trace = stitch_traces(sub_ranges, sub_traces)
            if len(stitched_range) > 0:
                mask = (np.array(stitched_range) >= start) & (np.array(stitched_range) <= end)
                if np.any(mask):
                    trimmed_range = np.array(stitched_range)[mask].tolist()
                    trimmed_trace = stitched_trace[mask]
                    if len(trimmed_range) > 0:
                        small_subsegment_ranges.append(trimmed_range)
                        small_subsegment_traces.append(trimmed_trace)
                        if whisker_array is not None:
                            ax2.plot(trimmed_range, trimmed_trace, color=f'C{i%10}', 
                                    label=f'Small {i}' if i < 10 else None)

    # If no small subsegments were created, fall back to original method
    if not small_subsegment_ranges:
        if whisker_array is not None:
            plt.close(fig)
        return stitch_traces(window_ranges, window_traces)

    if not use_medium_level:
        final_range, final_trace = stitch_traces(small_subsegment_ranges, small_subsegment_traces)
        if whisker_array is not None:
            ax3.plot(final_range, final_trace, 'k-', linewidth=2, label='Final Stitched')
            if len(window_ranges) <= 10:
                ax1.legend(loc='upper right')
            if len(small_windows) <= 10:
                ax2.legend(loc='upper right')
            ax3.legend(loc='upper right')
            plt.tight_layout()
            plt.show()
        return final_range, final_trace

    # Second Level: Split into medium overlapping windows
    medium_windows = []
    start = min_time
    while start < max_time:
        end = min(start + medium_window_size - 1, max_time)
        medium_windows.append((start, end))
        start += medium_step_size

    if whisker_array is not None:
        # Plot medium subsegment windows
        for i, (start, end) in enumerate(medium_windows):
            ax2.axvspan(start, end, alpha=0.1, color=f'C{(i+3)%10}')
            ax2.text(start + (end-start)/2, ax2.get_ylim()[0] + (ax2.get_ylim()[1]-ax2.get_ylim()[0])*0.1, 
                    f'M{i}', ha='center', va='bottom', fontsize=8)

    # Stitch small subsegments within each medium window
    medium_subsegment_ranges = []
    medium_subsegment_traces = []
    for i, (start, end) in enumerate(medium_windows):
        sub_ranges = []
        sub_traces = []
        for r, t in zip(small_subsegment_ranges, small_subsegment_traces):
            r_min = min(r) if r else start
            r_max = max(r) if r else end
            if r_max >= start and r_min <= end:
                sub_ranges.append(r)
                sub_traces.append(t)
        
        if sub_ranges:
            stitched_range, stitched_trace = stitch_traces(sub_ranges, sub_traces)
            if len(stitched_range) > 0:
                mask = (np.array(stitched_range) >= start) & (np.array(stitched_range) <= end)
                if np.any(mask):
                    trimmed_range = np.array(stitched_range)[mask].tolist()
                    trimmed_trace = stitched_trace[mask]
                    if len(trimmed_range) > 0:
                        medium_subsegment_ranges.append(trimmed_range)
                        medium_subsegment_traces.append(trimmed_trace)
                        if whisker_array is not None:
                            ax3.plot(trimmed_range, trimmed_trace, color=f'C{(i+3)%10}', 
                                    label=f'Medium {i}' if i < 10 else None)

    # If no medium subsegments were created, stitch small subsegments directly
    if not medium_subsegment_ranges:
        final_range, final_trace = stitch_traces(small_subsegment_ranges, small_subsegment_traces)
        if whisker_array is not None:
            ax3.plot(final_range, final_trace, 'k-', linewidth=2, label='Final Stitched')
            if len(window_ranges) <= 10:
                ax1.legend(loc='upper right')
            if len(small_windows) <= 10:
                ax2.legend(loc='upper right')
            ax3.legend(loc='upper right')
            plt.tight_layout()
            plt.show()
        return final_range, final_trace

    # Third Level: Stitch medium subsegments into final trace
    final_range, final_trace = stitch_traces(medium_subsegment_ranges, medium_subsegment_traces)
    
    if whisker_array is not None:
        ax3.plot(final_range, final_trace, 'k-', linewidth=2, label='Final Stitched')
        if len(window_ranges) <= 10:
            ax1.legend(loc='upper right')
        if len(small_windows) <= 10:
            ax2.legend(loc='upper right')
        if len(medium_windows) <= 10:
            ax3.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
    
    return final_range, final_trace



def normalize_all_peaks(peak_coords, mean_angle):
    normalized_peaks = []
    for t, peaks in enumerate(peak_coords):
        if len(peaks) > 0:
            normalized = [(p[0] - mean_angle[t], p[1]) for p in peaks]
        else:
            normalized = []
        normalized_peaks.append(normalized)
    return normalized_peaks


def track_peaks(normalized_peaks, max_inactive=5, max_distance=3, min_length=25, history_window=8, history_threshold=0.5):
    # Initialize tracks
    tracks = [Track(p[0], 0) for p in normalized_peaks[0]] if normalized_peaks[0] else []
    completed_tracelets = []
    
    # Track over time
    for t in range(1, len(normalized_peaks)):
        #print(t)
        current_peaks = normalized_peaks[t]
        if not current_peaks:
            # Vectorized operation for marking tracks inactive
            for track in tracks:
                track.mark_inactive(max_inactive, history_window, history_threshold)
            continue
        
        active_tracks = [track for track in tracks if track.active]
        if not active_tracks:
            # If no active tracks, create new ones for all peaks
            tracks.extend([Track(peak[0], t) for peak in current_peaks])
            continue
        
        # Predict positions - generate this once
        predicted = np.array([track.predict(time=t) for track in active_tracks])
        
        # Optimization: Create arrays of peak positions and values for vectorized operations
        peak_positions = np.array([p[0] for p in current_peaks])
        assigned_peaks = set()
        
        # Calculate all distances at once using broadcasting
        # This creates a matrix of distances between each track and each peak
        distances = np.abs(predicted[:, np.newaxis] - peak_positions)
        
        # For each track, find the closest unassigned peak
        for i, track in enumerate(active_tracks):
            track_distances = distances[i]
            
            # Find valid peaks (within max_distance)
            valid_indices = np.where(track_distances < max_distance)[0]

            if len(valid_indices) > 0:

                # Get distances for unassigned valid peaks
                assignable_distances = [track_distances[idx] for idx in valid_indices]
                # Find index of minimum distance
                min_idx = valid_indices[np.argmin(assignable_distances)]
                
                # Update track and mark peak as assigned
                track.update(peak_positions[min_idx], t)
                assigned_peaks.add(min_idx)

                if len(valid_indices) > 1:
                    # Get all unassigned indices at once
                    unassigned_indices = valid_indices[valid_indices != min_idx]
                    
                    # Create all new tracks at once
                    new_tracks = []
                    for idx in unassigned_indices:
                        new_track = track.duplicate()
                        new_track.positions[-1] = peak_positions[idx]
                        new_tracks.append(new_track)
                        assigned_peaks.add(idx)
                    
                    # Extend tracks list once
                    tracks.extend(new_tracks)
            else:
                track.mark_inactive(max_inactive, history_window, history_threshold)

        # Create new tracks for any unassigned peaks - vectorized with a list comprehension
        unassigned_indices = [j for j in range(len(current_peaks)) if j not in assigned_peaks]
        new_tracks = [Track(peak_positions[j], t) for j in unassigned_indices]
        tracks.extend(new_tracks)

        # Process inactive tracks efficiently using list comprehensions
        inactive_tracks = [track for track in tracks if not track.active and len(track.positions) >= min_length]
        completed_tracelets.extend(inactive_tracks)
        tracks = [track for track in tracks if track.active]

    # Add remaining tracks that meet the minimum length requirement
    valid_remaining_tracks = [track for track in tracks if len(track.positions) >= min_length]
    completed_tracelets.extend(valid_remaining_tracks)

    return completed_tracelets

def denormalize_tracelets(tracelets, mean_envelope):
    """
    Denormalize tracelets back to original coordinate space.
    
    Args:
        tracelets: List of Track objects with normalized positions
        mean_envelope: Array of mean envelope positions for each time point
        scale_factors: Array of scale factors used for normalization
        
    Returns:
        List of denormalized Track objects
    """
    denormalized_tracelets = []
    
    for tracelet in tracelets:
        # Create a copy of the tracelet to avoid modifying the original
        denormalized_track = tracelet.duplicate()
        
        # Denormalize each position
        for i, t in enumerate(tracelet.times):
            # Get the corresponding mean and scale factor for this time point
            mean_pos = mean_envelope[t]
            
            # Denormalize: original_pos = normalized_pos * scale + mean
            denormalized_track.positions[i] = tracelet.positions[i] + mean_pos
        
        denormalized_tracelets.append(denormalized_track)
    
    return denormalized_tracelets


def get_separated_tracelets(tracelets):
    separated_tracelets = []
    for i, tracelet in enumerate(tracelets):
        if i == 0:
            separated_tracelets.append(tracelet)
            tracelet_coords = [pos for pos in zip(tracelet.positions, tracelet.times)] 
        else:
            # check that the tracelet doesn't have any overlapping coordinates with the tracelet coords
            if not any(coord in tracelet_coords for coord in zip(tracelet.positions, tracelet.times)):
                separated_tracelets.append(tracelet)
                tracelet_coords.extend([pos for pos in zip(tracelet.positions, tracelet.times)])
    return separated_tracelets


def average_window_traces(separated_tracelets, vslice, window_size=4):
    slice_samples = []
    
    # Store all window traces for later stitching
    all_window_traces = []
    all_window_ranges = []
    
    # Process each possible window in the valid slice
    for window_start in range(vslice[0], vslice[1] - window_size + 2):
        window_end = window_start + window_size - 1
        window_range = range(window_start, window_end + 1)
        
        # Find tracelets that span the entire window
        full_coverage_tracelets = []
        for tracelet in separated_tracelets:
            # Check if tracelet covers the entire window
            has_start = window_start in tracelet.times or min(tracelet.times) < window_start
            has_end = window_end in tracelet.times or max(tracelet.times) > window_end
            
            # Only use tracelets that don't start or end within the window
            starts_before_window = min(tracelet.times) < window_start
            ends_after_window = max(tracelet.times) > window_end
            
            if has_start and has_end and starts_before_window and ends_after_window:
                # Get or interpolate positions for each timepoint in the window
                
                # Get existing times within the window
                existing_times = [t for t in tracelet.times if window_start <= t <= window_end]
                existing_positions = [tracelet.positions[tracelet.times.index(t)] for t in existing_times]
                
                if len(existing_times) >= 2:  # Need at least 2 points for interpolation
                    interp_func = interp1d(existing_times, existing_positions, 
                                        bounds_error=False, fill_value="extrapolate")
                    
                    # Generate interpolated positions for all points in the window
                    interp_positions = interp_func(list(window_range))
                    full_coverage_tracelets.append(interp_positions)
        
        # If we have tracelets with full coverage, calculate mean trace for this window
        if full_coverage_tracelets:
            # Calculate mean position at each timepoint in the window
            window_mean_trace = np.mean(full_coverage_tracelets, axis=0)
            
            # Save the window trace and its time range for later stitching
            all_window_traces.append(window_mean_trace)
            all_window_ranges.append(list(window_range))
            
            # Plot the mean trace for this window - REMOVED PLOTTING HERE
            # plt.plot(window_range, window_mean_trace, c='r', linewidth=1)

            # Save the number of tracelets used for this window
            window_samples = len(full_coverage_tracelets)
            slice_samples.append(window_samples)
        else:
            # If no valid tracelets for this window, record 0
            slice_samples.append(0)

    return all_window_traces, all_window_ranges, slice_samples


def average_lower_upper_window_traces(separated_tracelets, vslice, window_size=4, overlap_percent=50):
    upper_window_traces = []
    lower_window_traces = []
    all_window_ranges = []
    slice_samples = []
    
    # Process each possible window in the valid slice
    for window_start in range(vslice[0], vslice[1] - window_size + 2):
        window_end = window_start + window_size - 1
        window_range = range(window_start, window_end + 1)
        
        # Find tracelets that span the entire window
        full_coverage_tracelets = []
        for tracelet in separated_tracelets:
            # Check if tracelet covers the entire window
            has_start = window_start in tracelet.times or min(tracelet.times) < window_start
            has_end = window_end in tracelet.times or max(tracelet.times) > window_end
            
            # Only use tracelets that don't start or end within the window
            starts_before_window = min(tracelet.times) < window_start
            ends_after_window = max(tracelet.times) > window_end
            
            if has_start and has_end and starts_before_window and ends_after_window:
                # Get or interpolate positions for each timepoint in the window
                
                # Get existing times within the window
                existing_times = [t for t in tracelet.times if window_start <= t <= window_end]
                existing_positions = [tracelet.positions[tracelet.times.index(t)] for t in existing_times]
                
                if len(existing_times) >= 2:  # Need at least 2 points for interpolation
                    interp_func = interp1d(existing_times, existing_positions, 
                                        bounds_error=False, fill_value="extrapolate")
                    
                    # Generate interpolated positions for all points in the window
                    interp_positions = interp_func(list(window_range))
                    full_coverage_tracelets.append(interp_positions)
        
        # If we have tracelets with full coverage, calculate upper and lower mean traces
        if full_coverage_tracelets:
            # Calculate mean position of each tracelet over the window
            tracelet_means = [np.mean(trace) for trace in full_coverage_tracelets]
            
            # Sort tracelets by their mean position
            sorted_indices = np.argsort(tracelet_means)
            sorted_tracelets = [full_coverage_tracelets[i] for i in sorted_indices]
            
            # Determine split point for 60% overlap
            n_tracelets = len(sorted_tracelets)
            lower_count = int(n_tracelets * overlap_percent / 100)
            upper_count = lower_count            
            
            # Create the lower and upper groups with overlap
            lower_tracelets = sorted_tracelets[:lower_count]
            upper_tracelets = sorted_tracelets[-upper_count:]
            
            # Calculate mean traces for each group
            lower_window_mean = np.mean(lower_tracelets, axis=0)
            upper_window_mean = np.mean(upper_tracelets, axis=0)
            all_window_traces_mean = np.mean(full_coverage_tracelets, axis=0)
            if not np.isnan(lower_window_mean).any() and not np.isnan(upper_window_mean).any():
                # Save the window traces and range
                lower_window_traces.append(lower_window_mean)
                upper_window_traces.append(upper_window_mean)
                all_window_ranges.append(list(window_range))
            else:
                lower_window_traces.append(all_window_traces_mean)
                upper_window_traces.append(all_window_traces_mean)
                all_window_ranges.append(list(window_range))

                
            # Save number of tracelets
            slice_samples.append(n_tracelets)
        else:
            slice_samples.append(0)

    return lower_window_traces, upper_window_traces, all_window_ranges, slice_samples


def plot_normalized_peaks(normalized_peaks, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()
    
    # Create a colormap for visualization
    cmap = plt.cm.viridis
    
    # Plot each time point's peaks
    for t, peaks in enumerate(normalized_peaks):
        if len(peaks) > 0:
            # Extract y positions and pixel values
            y_positions = [p[0] for p in peaks]
            pixel_values = [p[1] for p in peaks]
            
            # Use pixel values to determine color intensity
            colors = [cmap(val/max(pixel_values)) for val in pixel_values]
            
            # Plot the peaks
            ax.scatter([t] * len(peaks), y_positions, c=colors, s=20, alpha=0.7)
    
    # Plot the mean angle (zero line after normalization)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Mean Envelope')
    
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Normalized Position')
    ax.invert_yaxis()
    ax.legend()

    return fig, ax


def plot_normalized_traces(normalized_peaks, completed_tracelets, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    # Plot the normalized peaks
    fig, ax = plot_normalized_peaks(normalized_peaks, ax)

    # Plot the completed traces
    for trace in completed_tracelets:
        plt.plot(trace.times, trace.positions, linewidth=2)
    return fig, ax


def plot_denormalized_tracelets(whisk_arr, denormalized_tracelets, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()
        
    # Plot the whisker array
    ax.imshow(whisk_arr, cmap='gray')

    # Plot the denormalized tracelets
    for trace in denormalized_tracelets:
        ax.plot(trace.times, trace.positions, linewidth=2)
    return fig, ax

# Function to normalize tracelets relative to a setpoint trace
def normalize_tracelets_to_setpoint(tracelets, setpoint_trace):
    relative_tracelets = []
    for tracelet in tracelets:
        relative_track = copy.deepcopy(tracelet) # Use deepcopy to avoid modifying original
        new_positions = []
        new_times = []
        for i, t in enumerate(tracelet.times):
            if 0 <= t < len(setpoint_trace) and not np.isnan(setpoint_trace[t]):
                relative_pos = tracelet.positions[i] - setpoint_trace[t]
                new_positions.append(relative_pos)
                new_times.append(t)
            # Else: Skip point if setpoint is NaN or time out of bounds

        # Update track with valid relative points
        relative_track.positions = new_positions
        relative_track.times = new_times
        if relative_track.times: # Only add if track still has points
            relative_tracelets.append(relative_track)
    return relative_tracelets

# Function to separate tracelets based on their mean position relative to zero
def separate_relative_tracelets(relative_tracelets, threshold=0):
    upper_tracelets = []
    lower_tracelets = []
    for tracelet in relative_tracelets:
        if tracelet.positions: # Ensure there are positions to calculate mean
            mean_pos = np.nanmean(tracelet.positions)
            if mean_pos > threshold:
                upper_tracelets.append(tracelet)
            else:
                lower_tracelets.append(tracelet)
    print(f"Separated into {len(upper_tracelets)} upper and {len(lower_tracelets)} lower tracelets.")
    return upper_tracelets, lower_tracelets