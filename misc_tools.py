# miscellaneous functions that are needed for whisking envelope extraction

import numpy as np
from scipy.signal import savgol_filter
import math
import cv2


def first_non_nan(a_list):
    """
    Find the first non-NaN value in a list.
    Can also be used to find the last non-NaN value by reversing the list (a_list[::-1]) 
    Parameters
    ----------
    a_list : list of values
    Returns
    -------
    i : index of first non-NaN value
    """
    
    for i, item in enumerate(a_list):
    
        if np.isnan(item) == False:
        
            return i


def first_non_empty(a_list):
    """
    Find the first non-empty list in a list.
    Can also be used to find the last non-empty list by reversing the list (a_list[::-1]) 
    Parameters
    ----------
    a_list : list of values
    Returns
    -------
    i : index of first non-empty list
    """
    
    for i, item in enumerate(a_list):
    
        if len(item) > 0:
        
            return i


def first_index_equal(a_list, value):
    
    for i, item in enumerate(a_list):
    
        if item == value:
            
            return i


def first_index_above(a_list, value, threshold):
    
    for i, item in enumerate(a_list):
    
        if item >= value and item < threshold:
        
            return i
        

def first_index_under(a_list, value, threshold):
    
    for i, item in enumerate(a_list):
    
        if item <= value and item > threshold:
            
            return i


def detect_contiguity(List,threshold = None):
    _List = np.asarray(List.copy())

    if threshold is not None :
        if np.isnan(threshold) :
            for idx , val in enumerate(_List) :
                if not np.isnan(val):
                    _List[idx] = 1
        else :
            for idx , val in enumerate(_List) :
                if not np.isnan(val) and val >= threshold :
                    _List[idx] = 1
                if not np.isnan(val) and val < threshold :
                    _List[idx] = 0

    ranges = [i+1  for i in range(len(_List[1:])) if not ( ( _List[i] == _List[i+1] ) or ( math.isnan(_List[i]) and math.isnan(_List[i+1]) ) ) ]
    ranges.append(len(_List))
    ranges.insert(0, 0)

    slices = []
    values = []

    for i in range(len(ranges)-1):
        slices.append([ranges[i], ranges[i+ 1]])
        if _List[ranges[i]] is None :
            values.append(None)
        else :
            values.append(_List[ranges[i]])

    return slices, values


def euclid2(x,y):
    """
    Get Euclidean Distance between 2 points
    Parameters
    ----------
    x : tuple or list
    y : tuple or list
    
    Returns
    -------
    returns a value (the euclidean distance)
    the euclidean distance is defined as: (x1 – y1)**2 + (x2 – y2)**2 + (xn – yn)**2 +.....
    
    """
    return sum((xi-yi)**2 for xi,yi in zip(x,y))


def slope(x1, y1, x2, y2):
    """
    Given 2 xy coordinates, calculate the slope
    Parameters
    ----------
    x1 : x coordinate point 1
    y1 : y coordinate point 1
    x2 : x coordinate point 2
    y2 : y coordinate point 2
    Returns
    -------
    m : value
        slope between xy1 and xy2, defined as: (y2-y1)/(x2-x1)
    """
    m = (y2-y1)/(x2-x1)
    return m  


def intercept(x1, y1, m):
    """
    Given 1 xy coordinate and the slope, calculate the intercept
    Parameters
    ----------
    x1 : x coordinate point 1
    y1 : y coordinate point 1
    m : slope
    Returns
    -------
    b : value
        intercept of function going through xy1 with slope m
        defined as: y1 - m * x1
    """
    b = y1 - m * x1
    return b


def savgol_smooth(value_arr, window_size=101):
    """
    Smooth any list of values using a savgol filter (order 3)
    Can also contain nans (nans will obviously not be smoothed)

    Parameters
    ----------
    value_arr : list
    window_size : integer (needs to be an uneven number), optional
        window size for using our savgol filter. The default is 101.

    Returns
    -------
    smoothed_array : list
        savgol-smoothed version of input list

    """

    shape = len(value_arr)
    smoothed_array = np.zeros(shape)
    smoothed_array[:] = np.nan

    slices, values = detect_contiguity(value_arr, np.nan)
    valid_slices = [slices for i, slices in enumerate(slices) if values[i]==1]

    for vslice in valid_slices:

        smoothed_data = savgol_filter(value_arr[vslice[0]:vslice[1]], window_size, 3, mode='nearest')
        smoothed_array[vslice[0]:vslice[1]] = smoothed_data

    return smoothed_array


def video_to_array(input_loc):
    """
    Function to extract frames from input video file
    and put them into an array.

    Parameters
    ----------
    input_loc : string
        Input video filepath

    Returns
    -------
    vid_arr : 3D array
        NR OF FRAMES x VID_Y x VID_X
    """
    
    vid_arr = []
    
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
 
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    count = 0
    
    # Start converting the video
    while cap.isOpened():
        
        # Extract the frame
        ret, frame = cap.read()   
        
        vid_arr.append(frame[:,:,0])
        count = count + 1
        
        # If there are no more frames left
        if (count > (video_length)):

            # Release the feed
            cap.release()
            break
            
    return vid_arr


def subsample_image(xs, ys, img):
    """
    Given a list of floating point coordinates (Nx2) in the image,
    return the pixel value at each location using bilinear interpolation.

    Parameters
    ----------
    xs : float or list of floats
    ys : float or list of floats
    img : 2D array, type = uint32 
    
    Returns
    -------
    weights : float or list of floats
        pixel value(s) at floating point coordinate(s)

    """
    if len(img.shape) == 2:
        img = np.expand_dims(img, 2)

    pxs = np.floor(xs).astype(int)
    pys = np.floor(ys).astype(int)
    dxs = xs-pxs
    dys = ys-pys
    wxs, wys = 1.0-dxs, 1.0-dys
    
    weights =  np.multiply(img[pys, pxs, :].T      , wxs*wys).T
    weights += np.multiply(img[pys, pxs+1, :].T    , dxs*wys).T
    weights += np.multiply(img[pys+1, pxs, :].T    , wxs*dys).T
    weights += np.multiply(img[pys+1, pxs+1, :].T  , dxs*dys).T
    return weights