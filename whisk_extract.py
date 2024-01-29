import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import fsolve

import whisk_extract_legacy, misc_tools

# The code here is an updated version to identify and trace a ROI in front of the whiskerpad. 
# However it required the user to use something like DeepLabCut to track the whiskerpad contour.
# Overall, this is probably more accurate than the legacy version, but it requires more work from the user (creating a DLC project, training, tracking etc.).


def extract_whisker_video(vid_arr, nose_positions, head_directions, crop_size=100, side='R'):
    """
    Extract whisker video by cropping, rotating, and cutting each frame based on nose positions and head directions.
    
    Parameters
    ----------
    vid_arr : 3D numpy array
        Video represented as a series of frames.
    nose_positions : DataFrame
        Contains nose coordinates for each frame.
    head_directions : list or 1D array
        Direction of the head for each frame.
    crop_size : int, default = 100
        Defines the size of the image cropped around the nose position.
    side : str, default = 'R' (L or R)
        Defines which side of the mouse to extract the raw whisking pixel-values from.
        
    Returns
    -------
    cropped_vid : list
        List of processed frames.
    """
    
    cropped_vid = []

    for index, row in nose_positions.iterrows():
        if not np.isnan(head_directions[index]) and not np.isnan(row['nose_x']):
            # CUT, CROP, ROTATE
            cut_img, _, _ = whisk_extract_legacy.crop_rot_cut(vid_arr[index], row['nose_x'], row['nose_y'], crop_size, head_directions[index], side=side)
            cropped_vid.append(cut_img)
        else:
            # In case the condition isn't met, append an empty array.
            # The empty array will have the same shape as the original frame to keep the video's consistency.
            empty_frame = np.zeros(np.shape(vid_arr[index]))
            empty_frame[:] = np.nan
            cropped_vid.append(empty_frame)

    return cropped_vid


def calculate_average_whiskerpad_contour(all_coordinates_df, columns, plot=False):
    """
    Calculate the average contour of the whiskerpad. 
    
    Parameters
    ----------
    all_coordinates_df : DataFrame
        Contains all the coordinates of the whiskerpad for each frame of a session.
    columns : list
        List of column names of the DataFrame.

    Returns
    -------
    xs : list
        List of x coordinates of the average contour.
    ys : list
        List of y coordinates of the average contour.
    """

    xs = []
    ys = []
    for col in columns:
        x = all_coordinates_df[col+'_x']
        y = all_coordinates_df[col+'_y']
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        xs.append(mean_x)
        ys.append(mean_y)
    
    if plot:
        fig, ax = plt.subplots(figsize=(5, 5))
        colours = ['cyan', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'magenta', 'brown', 'red']
        for i, col in enumerate(columns):
            ax.scatter(all_coordinates_df[col+'_x'], all_coordinates_df[col+'_y'], alpha=0.1, color=colours[i], zorder=0)
        ax.invert_yaxis()
        ax.plot(xs, ys, color='black', zorder=1, linewidth=3)
        plt.show()

    return xs, ys


def reposition_whiskerpad_contour(avg_xs, avg_ys, new_x, new_y):
    """
    Reposition the average whiskerpad contour to the nose position.

    Parameters
    ----------
    avg_xs : list
        List of x coordinates of the average contour.
    avg_ys : list
        List of y coordinates of the average contour.
    new_x : float
        X coordinate of the nose position.
    new_y : float   
        Y coordinate of the nose position.

    Returns 
    -------
    new_xs : list
        List of x coordinates of the repositioned contour.
    new_ys : list   
        List of y coordinates of the repositioned contour.
    """

    x_diff = new_x - avg_xs[0]
    y_diff = new_y - avg_ys[0]
    new_xs = [x + x_diff for x in avg_xs]
    new_ys = [y + y_diff for y in avg_ys]

    return new_xs, new_ys


def rotate_whiskerpad_contour(xs, ys, nose_x, nose_y, rotation_angle):
    """
    Rotate the whiskerpad contour around the nose position.

    Parameters
    ----------
    xs : list
        List of x coordinates of the contour.
    ys : list   
        List of y coordinates of the contour.
    nose_x : float  
        X coordinate of the nose position.
    nose_y : float  
        Y coordinate of the nose position.
    rotation_angle : float  
        Angle of rotation in radians.

    Returns
    -------
    rotated_xs : list
        List of x coordinates of the rotated contour.
    rotated_ys : list
        List of y coordinates of the rotated contour.
    """
    rotated_xs = []
    rotated_ys = []
    for x, y in zip(xs, ys):
        # Translate points to origin for rotation
        x -= nose_x
        y -= nose_y
        # Rotate points
        rx, ry = misc_tools.rotate_around_point((x, y), rotation_angle)
        # Translate points back
        rx += nose_x
        ry += nose_y
        rotated_xs.append(rx)
        rotated_ys.append(ry)
    return rotated_xs, rotated_ys


def create_offset_parallel_line(xs, ys, slope, d=7):
    """
    Offset the whiskerpad contour to the upper left

    Parameters
    ----------
    xs : list
        List of x coordinates of the contour.
    ys : list
        List of y coordinates of the contour.
    slope : float
        Slope of the line.
    d : int, default = 7
        Distance to offset the contour.

    Returns
    -------
    offset_xs : list
        List of x coordinates of the offset contour.
    offset_ys : list
        List of y coordinates of the offset contour.
    """

    angle = np.arctan(slope)
    # Calculate offset in parallel to x and y directions
    offset_x = d * np.sin(angle)
    offset_y = d * np.cos(angle)

    offset_xs = [x + offset_x for x in xs]
    offset_ys = [y - offset_y for y in ys]
    
    return offset_xs, offset_ys


def new_designate_roi_line(offset_xs, offset_ys, rotated_xs, nose_x, nose_y, sample_res=150):
    """
    Designate a line to sample whiskerpad pixel values from.

    Parameters
    ----------
    offset_xs : list
        List of x coordinates of the offset contour.
    offset_ys : list
        List of y coordinates of the offset contour.
    rotated_xs : list
        List of x coordinates of the rotated contour (before offset).
    nose_x : float
        X coordinate of the nose position.
    nose_y : float  
        Y coordinate of the nose position.
    sample_res : int, default = 150 
        Number of points to sample along the whiskerpad ROI.

    Returns 
    -------
    roi_xs : list
        List of x coordinates of the ROI.
    roi_ys : list
        List of y coordinates of the ROI.
    """
    mymodel = np.poly1d(np.polyfit(offset_xs, offset_ys, 2))
    # Define the target y value
    target_y = nose_y
    # Define a function that calculates the difference between the model's prediction and the target y value
    def difference_from_target_y(x):
        return mymodel(x) - target_y
    # Provide an initial guess for x. It's good to choose a value that's within the range of your data.
    initial_guess = nose_x
    # Use fsolve to find the x value where the difference function is zero
    x_solution = fsolve(difference_from_target_y, initial_guess)
    myline = np.linspace(x_solution, rotated_xs[-1], sample_res)
    roi_xs = []
    roi_ys = []
    for x in myline:
        y = mymodel(x)
        roi_xs.append(list(x)[0])
        roi_ys.append(list(y)[0])
    return roi_xs, roi_ys


def new_extract_raw_whisk(whisker_video_path, avg_xs, avg_ys, tracking_data, columns, offset_dist=7, sample_res=150, plot=False):

    """
    Extract raw whiskerpad pixel values from the whisker video.
    
    Parameters
    ----------
    whisker_video_path : string
        Path to the whisker video.
    avg_xs : list
        List of x coordinates of the average contour.
    avg_ys : list
        List of y coordinates of the average contour.
    tracking_data : DataFrame
        Contains whisker pad contour coordinates for each frame.
        Needs to contain a coordinate for the nose position (nose_x, nose_y)
    columns : list
        List of column names of the DataFrame.
        Needs to start with the nose
    offset_dist : int, default = 7  
        Distance to offset the whiskerpad contour.
    sample_res : int, default = 150
        Number of points to sample along the whiskerpad ROI.
    plot : bool, default = False
        If True, plots the raw whiskerpad pixel values.
        
    Returns
    -------
    raw_whisk : list
        List of raw whiskerpad pixel values.
    """

    video_array = misc_tools.video_to_array(whisker_video_path)
    slope, _, _, _, _ = stats.linregress(avg_xs, avg_ys)

    raw_whisking_vals = np.empty((len(video_array), sample_res))
    raw_whisking_vals[:] = np.nan

    for index, row in tracking_data.iterrows():
        # check that there are no NaN values in any of the columns (both x and y)
        if not np.isnan(row).any() and np.sum(video_array[int(index)]) > 4:
            new_xs, new_ys = reposition_whiskerpad_contour(avg_xs, avg_ys, row['nose_x'], row['nose_y'])
            
            target_slope, _, _, _, _ = stats.linregress([row[column+'_x'] for column in columns], [row[column+'_y'] for column in columns])
            initial_angle = np.arctan(slope) # convert slope to radians
            target_angle = np.arctan(target_slope) # convert slope to radians
            rotation_angle = target_angle - initial_angle # Calculate angle of rotation

            rotated_xs, rotated_ys = rotate_whiskerpad_contour(new_xs, new_ys, row['nose_x'], row['nose_y'], rotation_angle)

            offset_xs, offset_ys = create_offset_parallel_line(rotated_xs, rotated_ys, target_slope, d=offset_dist)

            roi_xs, roi_ys = new_designate_roi_line(offset_xs, offset_ys, rotated_xs, row['nose_x'], row['nose_y'], sample_res=sample_res)

            pixel_vals = np.squeeze(misc_tools.subsample_image(roi_xs, roi_ys, video_array[int(index)]))
            raw_whisking_vals[int(index)] = pixel_vals

    if plot:
        plt.figure()
        plt.title('Raw Whisker-Pixel Values')
        plt.imshow(raw_whisking_vals.T)
        plt.xlabel('Time')
        plt.ylabel('Whiskerpad ROI')
        plt.show()
    
    return raw_whisking_vals.T