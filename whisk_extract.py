import numpy as np

import whisk_extract_legacy


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