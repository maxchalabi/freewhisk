import sys, os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from itertools import product
import whisk_params, whisk_tools

# Add your packages path
sys.path.append(r'C:\Users\Max\Documents\Python\__packages__')

# Import your libraries
import mLabAna as mana

# Load session data
SD = mana.SessionData(4199)

# Find the first available WhiskerTracking_Results
for index, row in SD.iterrows():
    if type(row['WhiskerTracking_Results']) == np.ndarray and index == 100:
        whisker_results = row['WhiskerTracking_Results']
        break

# Function to extract whisking envelope with given parameters
def extract_envelope_with_params(whisker_data, prominence, bin_size, pixel_scale, cutoff=4, mean_setpoint=60):
    """Extract whisking envelope with specific parameters."""
    peak_coords, raw_img = whisk_tools.collect_peaks(whisker_data, prominence=prominence, cutoff=cutoff, pixel_scale=pixel_scale)
    thresh = whisk_tools.peak_thresh(peak_coords)
    mmp_slices = whisk_tools.track_mmp_slices(peak_coords, thresh, slice_size=bin_size)
    patched_mmp_slices = whisk_tools.patch_mmp_slices(mmp_slices, mean_setpoint, raw_img)
    envelope = whisk_tools.average_mmp_slices(patched_mmp_slices, (0, np.shape(whisker_data)[1]))
    
    # Calculate envelope quality metrics
    non_nan_ratio = np.sum(~np.isnan(envelope)) / len(envelope)
    if non_nan_ratio > 0:
        smoothness = np.nanmean(np.abs(np.diff(envelope, 2)))
        continuity = np.sum(np.isnan(envelope[1:]) != np.isnan(envelope[:-1]))
    else:
        smoothness = np.nan
        continuity = np.nan
        
    return envelope, non_nan_ratio, smoothness, continuity

# Create a parameter space to explore
prominence_values = [0.01, 0.05, 0.1, 0.5, 1.0]
bin_size_values = [3, 5, 7, 9]
pixel_scale_values = [0.5, 1]

# Display original data for reference
plt.figure(figsize=(12, 8))
plt.imshow(whisker_results)
plt.title('Original Whisker Tracking Results')
plt.colorbar(label='Pixel Intensity')
plt.show()

# Parameter Grid Exploration for all combinations
print("Running parameter grid exploration. This may take a moment...")

# Create figures for different pixel_scale values
for pixel_scale in pixel_scale_values:
    fig, axes = plt.subplots(len(prominence_values), len(bin_size_values), 
                            figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle(f'Whisking Envelopes: Pixel Scale = {pixel_scale}', fontsize=16)
    
    # Add column headers (bin sizes)
    for i, bin_size in enumerate(bin_size_values):
        axes[0, i].set_title(f'Bin Size = {bin_size}')
    
    # Add row labels (prominence values)
    for i, prominence in enumerate(prominence_values):
        axes[i, 0].set_ylabel(f'Prominence = {prominence}', fontsize=10)
    
    # Generate all envelopes for this pixel_scale value
    for i, prominence in enumerate(prominence_values):
        for j, bin_size in enumerate(bin_size_values):
            # Extract envelope and quality metrics
            envelope, non_nan_ratio, smoothness, continuity = extract_envelope_with_params(
                whisker_results, prominence, bin_size, pixel_scale)
            
            # Plot the envelope
            axes[i, j].imshow(whisker_results, aspect='auto', alpha=0.7)
            
            # Check if the envelope has data before plotting
            if non_nan_ratio > 0:
                axes[i, j].plot(range(len(envelope)), envelope, 'r-', linewidth=1.5)
                # Add quality metrics text
                axes[i, j].text(10, 10, f'Coverage: {non_nan_ratio:.2f}\nSmooth: {smoothness:.2f}', 
                               fontsize=7, color='white', backgroundcolor='black')
            else:
                axes[i, j].text(10, 10, "No valid envelope", fontsize=7, 
                               color='white', backgroundcolor='black')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Now create a quantitative comparison table
print("\nParameter Comparison Table (Coverage = higher is better, Smoothness = lower is better)")
print("="*80)
print(f"{'Prominence':<10} {'Bin Size':<10} {'Pixel Scale':<12} {'Coverage':<10} {'Smoothness':<12} {'Continuity':<10}")
print("-"*80)

# Store results for later sorting
results = []

# Generate summary metrics for all parameter combinations
for prominence, bin_size, pixel_scale in product(prominence_values, bin_size_values, pixel_scale_values):
    _, coverage, smoothness, continuity = extract_envelope_with_params(
        whisker_results, prominence, bin_size, pixel_scale)
    
    results.append((prominence, bin_size, pixel_scale, coverage, smoothness, continuity))
    print(f"{prominence:<10.2f} {bin_size:<10d} {pixel_scale:<12.2f} {coverage:<10.2f} {smoothness:<12.2f} {continuity:<10.0f}")

# Find the best parameter combinations (sort by coverage, then by smoothness)
print("\nTop 5 Parameter Combinations (sorted by coverage, then by smoothness):")
print("="*80)

# Sort results by coverage (descending) and then by smoothness (ascending)
sorted_results = sorted(results, key=lambda x: (-x[3], x[4]))

for i, (prominence, bin_size, pixel_scale, coverage, smoothness, continuity) in enumerate(sorted_results[:5]):
    print(f"{i+1}. Prominence={prominence}, Bin Size={bin_size}, Pixel Scale={pixel_scale} " + 
          f"(Coverage: {coverage:.2f}, Smoothness: {smoothness:.2f}, Breaks: {continuity:.0f})")

# Plot the top 3 parameter combinations
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle('Top 3 Parameter Combinations', fontsize=16)

for i, (prominence, bin_size, pixel_scale, _, _, _) in enumerate(sorted_results[:3]):
    # Extract envelope with these parameters
    envelope, _, _, _ = extract_envelope_with_params(
        whisker_results, prominence, bin_size, pixel_scale)
    
    # Plot on the original image
    axes[i].imshow(whisker_results, aspect='auto')
    axes[i].plot(range(len(envelope)), envelope, 'r-', linewidth=2)
    axes[i].set_title(f"#{i+1}: Prominence={prominence}, Bin Size={bin_size}, Pixel Scale={pixel_scale}")
    axes[i].set_ylabel('Pixel Position')

axes[2].set_xlabel('Frame')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Function to interactively test parameters
def test_parameters(whisker_data, prominence, bin_size, pixel_scale):
    """Run a full test with custom parameters and show all results."""
    # Extract the envelope
    peak_coords, raw_img = whisk_tools.collect_peaks(whisker_data, prominence=prominence, 
                                          cutoff=4, pixel_scale=pixel_scale)
    thresh = whisk_tools.peak_thresh(peak_coords)
    mmp_slices = whisk_tools.track_mmp_slices(peak_coords, thresh, slice_size=bin_size)
    patched_mmp_slices = whisk_tools.patch_mmp_slices(mmp_slices, 60, raw_img)
    envelope = whisk_tools.average_mmp_slices(patched_mmp_slices, (0, np.shape(whisker_data)[1]))
    
    # Create a figure with 4 subplots showing the process
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    
    # Plot 1: Original image with detected peaks
    axes[0].imshow(whisker_data, aspect='auto')
    axes[0].set_title(f'Detected Peaks (prominence={prominence}, pixel_scale={pixel_scale})')
    
    # Plot peaks on the image
    for i, peaks in enumerate(peak_coords):
        if peaks:
            y_positions = [p[0] for p in peaks]
            axes[0].scatter([i] * len(y_positions), y_positions, c='r', s=2)
    
    # Plot 2: Tracked peak movement
    axes[1].imshow(whisker_data, aspect='auto')
    axes[1].set_title(f'Tracked Peak Movement (bin_size={bin_size})')
    
    # Plot tracked peaks
    x_vals = []
    y_vals = []
    for i, slice_data in enumerate(mmp_slices):
        if slice_data:
            for time, y_pos in slice_data:
                x_vals.append(time)
                y_vals.append(y_pos)
    
    if x_vals:
        axes[1].scatter(x_vals, y_vals, c='g', s=4, alpha=0.7)
    
    # Plot 3: Patched data
    axes[2].imshow(whisker_data, aspect='auto')
    axes[2].set_title('After Gap Patching')
    
    # Draw patched data as small line segments
    for i, slice_data in enumerate(patched_mmp_slices):
        if slice_data:
            x_vals = [t for t, _ in slice_data]
            y_vals = [y for _, y in slice_data]
            axes[2].plot(x_vals, y_vals, 'g-', linewidth=0.8, alpha=0.7)
    
    # Plot 4: Final envelope
    axes[3].imshow(whisker_data, aspect='auto')
    axes[3].plot(range(len(envelope)), envelope, 'r-', linewidth=2)
    axes[3].set_title('Final Whisking Envelope')
    
    # Add metrics to title
    non_nan_ratio = np.sum(~np.isnan(envelope)) / len(envelope)
    
    if non_nan_ratio > 0:
        smoothness = np.nanmean(np.abs(np.diff(envelope, 2)))
        continuity = np.sum(np.isnan(envelope[1:]) != np.isnan(envelope[:-1]))
        fig.suptitle(f'Parameters: prominence={prominence}, bin_size={bin_size}, pixel_scale={pixel_scale}\n' +
                    f'Coverage: {non_nan_ratio:.2f}, Smoothness: {smoothness:.2f}, Breaks: {continuity}', 
                    fontsize=14)
    else:
        fig.suptitle(f'Parameters: prominence={prominence}, bin_size={bin_size}, pixel_scale={pixel_scale}\n' +
                    'No valid envelope detected', fontsize=14)
    
    # Set labels for all plots
    for ax in axes:
        ax.set_ylabel('Pixel Position')
    
    axes[3].set_xlabel('Frame')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    return envelope

# Print instructions for using the test_parameters function
print("\nTo test your own parameter combinations, run:")
print("envelope = test_parameters(whisker_results, prominence=0.xx, bin_size=y, pixel_scale=zzz)")
print("Replace the values with your preferred parameters based on the results above.")

# Test the best combination from our analysis
best_prominence, best_bin_size, best_pixel_scale = sorted_results[0][:3]
best_envelope = test_parameters(whisker_results, best_prominence, best_bin_size, best_pixel_scale)

# Print final message
print(f"\nRecommended parameters based on analysis:")
print(f"- prominence: {best_prominence}")
print(f"- bin_size: {best_bin_size}")
print(f"- pixel_scale: {best_pixel_scale}")
print("\nYou can now extract metrics using these optimal parameters:")
print("amplitudes, max_amplitudes, whisk_speeds, pro_or_ret = fw.compute_trial_amps_and_speeds(best_envelope)")
print("phases, frequencies = fw.compute_trial_phases_and_freqs(best_envelope)") 