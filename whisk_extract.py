############################################################
############################################################
############################################################
#
#This Library here serves to extract the raw whisker pixel values over the course of a trial.
#This is done by finding the whiskerpad contour and designating a region of interest line 
#just in front of the whiskerpad in each frame of the video.
#
############################################################
############################################################
############################################################

import numpy as np
import cv2
import math
from  skimage import transform, feature
from scipy import signal
import sys
import matplotlib.pyplot as plt
import warnings

import misc_tools

warnings.simplefilter("ignore", category=RuntimeWarning)


def calculate_rotated_points(img, rot_img, angle, padding_vals, offset_val = 8):
    """
    It was useful/necessary to calculate the new position of the top image corners once the image is rotated
    This function does exactly that but with an offset

    Parameters
    ----------
    img : 2D image array
        Cropped and (potentially) padded image frame
    rot_img : 2D image array
        Rotated image
    angle : value
        degree by which cropped/padded image was rotated
    padding_vals : list of 2 tuples
        amount by which cropped image had been padded with 0 values
        in case that image was cropped at edges but we still want the nose in the middle
    offset_val : int, default = 8
        offset value to be added to the rotated point coordinates.
        The offset is necessary to make sure we really get rid of detected contours due to image rotation later on

    Returns
    -------
    xy1 : new xy coordinate of rotated top left img corner (+offset)
    xy2 : new xy coordinate of rotated top right img corner (+offset)

    """
    
    p1 = (0+padding_vals[0][0], 0+padding_vals[1][0]-padding_vals[1][1])
    p2 = (0+padding_vals[0][0], np.shape(img)[1]-1-padding_vals[1][1])
    
    cx = np.shape(img)[0]/2
    cy = np.shape(img)[1]/2
    
    xrot1=math.cos(math.radians(angle))*(p1[0]-cx)-math.sin(math.radians(angle))*(p1[1]-cy)+cx 
    yrot1=math.sin(math.radians(angle))*(p1[0]-cx)+math.cos(math.radians(angle))*(p1[1]-cy)+cy
    
    xrot2=math.cos(math.radians(angle))*(p2[0]-cx)-math.sin(math.radians(angle))*(p2[1]-cy)+cx 
    yrot2=math.sin(math.radians(angle))*(p2[0]-cx)+math.cos(math.radians(angle))*(p2[1]-cy)+cy
    
    xrot1 = xrot1 + np.shape(rot_img)[1]/2 - np.shape(img)[1]/2
    yrot1 = yrot1 + np.shape(rot_img)[0]/2 - np.shape(img)[0]/2
    
    xrot2 = xrot2 + np.shape(rot_img)[1]/2 - np.shape(img)[1]/2
    yrot2 = yrot2 + np.shape(rot_img)[0]/2 - np.shape(img)[0]/2
    
    return [yrot1+offset_val, xrot1+offset_val], [yrot2-offset_val, xrot2+offset_val]


def crop_rot_cut(img_arr, nose_x, nose_y, crop_size, angle, side = 'R', offset_val = 8):
    """
    Crop, Rotate and Cut a video frame to get the whiskerpad in view

    Parameters
    ----------
    img_arr : 2D numpy array
        video frame
    nose_x : value
        nose x coordinate
    nose_y : value
        nose y coordinate
    crop_size : int, default = 100
        Defines the size of the image cropped around the nose position
    angle : value
        degree by which img_arr should be rotated
    side : str, default = 'R' (L or R)
        Defines which side of the mouse to extract the raw whisking pixel-values from
    offset_val : int, default = 8
        offset value to be added to the rotated point coordinates.
        The offset is necessary to make sure we really get rid of detected contours due to image rotation later on

    Returns
    -------
    final_img : cropped, padded (if nose at edge of image), rotated and cut image 
    p1 : new xy coord of top left image corner
    p2 : new xy coord of top right image corner

    """
    
    xmin_crop = min(int(nose_x), crop_size)
    ymin_crop = min(int(nose_y), crop_size)
    xmax_crop = min(int(np.shape(img_arr)[1]-nose_x), crop_size)
    ymax_crop = min(int(np.shape(img_arr)[0]-nose_y), crop_size)
    
    #CROP
    cropped_img = img_arr[int(nose_y-ymin_crop):int(nose_y+ymax_crop), int(nose_x-xmin_crop):int(nose_x+xmax_crop)]
    
    #PAD
    padding_vals = [(crop_size-ymin_crop, crop_size-ymax_crop), (crop_size-xmin_crop, crop_size-xmax_crop)]
    padded_img = np.pad(cropped_img, padding_vals, mode='constant')
    #ROTATE
    rotated_img = transform.rotate(padded_img, angle, resize=True)
    
    p1, p2 = calculate_rotated_points(padded_img, rotated_img, angle, padding_vals, offset_val = offset_val) 
    
    #CUT & FIND NEW IMAGE BORDERS
    if side == 'R':
        final_img = rotated_img[:int(np.shape(rotated_img)[0]/2), :]
    if side == 'L':
        final_img = rotated_img[int(np.shape(rotated_img)[0]/2):, :][::-1]
        
    return final_img, p1, p2


def detect_edges(img_arr):
    """
    Detect edges using the canny filter.

    Parameters
    ----------
    img_arr : 2D numpy array
        cropped, padded (if nose at edge of image), rotated and cut image 

    Returns
    -------
    binary_arr : canny filter output but True values are set to 255.

    """
    
    blur_img = cv2.GaussianBlur(img_arr,(5,5),cv2.BORDER_DEFAULT)
    edges = feature.canny(blur_img, sigma=5).astype(int)
    binary_arr = np.where(edges==1, 255, edges)
    
    return binary_arr


def mask_edge_image(cut_img, edge_img, p1, p2):
    """
    This function serves to mask edges that have been detected with the canny edge filter
    but are due the rotation of the image.

    Parameters
    ----------
    cut_img : 2D image array (after crop_rot_cut)
    edge_img : 2D image array (after detect_edges)
    p1 : new xy coord of top left image corner
    p2 : new xy coord of top right image corner

    Returns
    -------
    masked_edges : 2D image array
        canny output put with masked parts set to 0
    mask : the mask we used to set values to 0

    """
    
    pad_val = 8
    
    b_idxs = [i for i, pixel in enumerate(cut_img[-1]) if pixel > 0]
    
    square = [np.array([p1, p2, [b_idxs[-pad_val], len(cut_img) - pad_val], [b_idxs[pad_val], len(cut_img) - pad_val]], dtype=np.int32)]
    
    hull_list = []
    for i in range(len(square)):
        hull = cv2.convexHull(square[i])
        hull_list.append(hull)
        
    poly_img = cv2.fillPoly(cut_img.copy(), hull_list, color=(255,255,255))
    masked_img = np.ma.masked_less(cv2.inRange(poly_img, 100, 255), 100)
    
    masked_edges = apply_mask(np.ma.getmask(masked_img), edge_img)

    return masked_edges, np.ma.getmask(masked_img)


def apply_mask(mask, edges):
    """
    Uses a masking array to set values in canny edge output to 0

    Parameters
    ----------
    mask : 2D boolen array
    edges : 2D array canny filter output, with Trues set to 255

    Returns
    -------
    pixel_arr : 2D array
        edge array with mask applied to it.

    """
    
    if isinstance(mask, np.bool_):
        mask = np.full((np.shape(edges)[0], np.shape(edges)[1]), False)
    
    pixel_arr = np.zeros((np.shape(edges)[0], np.shape(edges)[1]))
    for i, (m, v) in enumerate(zip(mask, edges)):
        for j, (n, w) in enumerate(zip(m, v)):
            if n == True:
                pass
            else:
                if w == 255:
                    pixel_arr[i, j] = 255
    
    return pixel_arr


def blur_edges(masked_edges):
    """
    To improve the detection of the full whiskerpad contour it is useful to blur the canny filter output.
    This function does this using cv2's GaussianBlur

    Parameters
    ----------
    masked_edges : 2D array output from mask_edge_image()

    Returns
    -------
    thresh_arr : blurred 2D edge array

    """
    
    blurred_edges = cv2.GaussianBlur(masked_edges,(5,5),cv2.BORDER_DEFAULT).astype(int)
    thresh_arr = cv2.inRange(blurred_edges, 1, 255)
    
    return thresh_arr


def find_edge_segments(masked_edges):
    """
    Detect individual contours using cv2.findContours.
    Also unifies contours that are close enough to each other

    Parameters
    ----------
    masked_edges : 2D blurred edge array

    Returns
    -------
    contours : array of arrays
        each array contains the points belonging to a contour

    """
    
    contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 1:
        contours = unify_contours(contours, 5)
        
    return contours


def find_if_close(cnt1,cnt2, max_dist):
    """
    Find if 2 contours are close to each other

    Parameters
    ----------
    cnt1 : array with points of first contour
    cnt2 : array with points of second contour
    max_dist : maximum distance for contours to be considered close enough to be unified

    Returns
    -------
    bool
        if true, contours are close, if False contours aren't close

    """
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < max_dist:
                return True
            elif i==row1-1 and j==row2-1:
                return False
            
            
def unify_contours(contours, max_dist):
    """
    Unify contours that are close to each other

    Parameters
    ----------
    contours : array of arrays
        each array contains the points belonging to a contour
    max_dist : maximum distance for contours to be considered close enough to be unified

    Returns
    -------
    unified : array of arrays
        each array contains the points belonging to a contour

    """
    
    LENGTH = len(contours)
    status = np.zeros((LENGTH,1))

    for i,cnt1 in enumerate(contours):
        x = i    
        if i != LENGTH-1:
            for j,cnt2 in enumerate(contours[i+1:]):
                x = x+1
                dist = find_if_close(cnt1,cnt2, max_dist)
                if dist == True:
                    val = min(status[i],status[x])
                    status[x] = status[i] = val
                else:
                    if status[x]==status[i]:
                        status[x] = i+1

    unified = []
    maximum = int(status.max())+1
    for i in range(maximum):
        pos = np.where(status==i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            unified.append(cont)
            
    return unified


def select_contour(contours, final_img):
    """
    From the contours we detected, select the contour that belongs to the whiskerpad.
    This is done based on multiple conditions (and if necessary very basic shapematching)

    Parameters
    ----------
    contours : array of arrays
        each array contains the points belonging to a contour
    final_img : 2D array from which we detected contours

    Returns
    -------
    contours : array of arrays
        should only contain a single array (the whiskerpad contour)

    """
    if len(contours) > 1:
        contours = [cont for cont in contours if len(cont) > np.shape(final_img)[1]/2]
        if len(contours) > 1:
            # we only select contours whose minimum x value bigger than half the image length minus 1/6th of the image length. 
            # In practice this means that we only select contours that don't points on the left of the image (minus 1/15th of the image length)
            contours = [cont for cont in contours if min(cont[:,:,0]) > np.shape(final_img)[1]/2 - np.shape(final_img)[1]/15]
            if len(contours) > 1:
                # we only select contours whose maximum y value is bigger than the image height/2.5. 
                # In practice this means that we only select contours that contain points on the lower half (actually 2.5) of the image
                contours = [cont for cont in contours if max(cont[:,:,1]) > np.shape(final_img)[0]/2.5]
                if len(contours) > 1:
                    contours = shapematch_contours(contours, final_img)

    return contours


def shapematch_contours(contours, final_img):
    """
    Select the contour that most closely resembles a line going from the nose diagonally to 
    the right at 1/4 of the image height (0 image height is at the very top)

    Parameters
    ----------
    contours : array of arrays
        each array contains the points belonging to a contour
    final_img : 2D array from which we detected contours

    Returns
    -------
    final_cont : array of arrays
        only contains a single array (the whiskerpad contour)

    """
    x1 = np.shape(final_img)[1]/2
    y1 = np.shape(final_img)[0] - 1
    x2 = np.shape(final_img)[1] - 1
    y2 = np.shape(final_img)[0]/4

    p1=np.array([x1,y1])
    p2=np.array([x2,y2])

    cont_dists = []
    for cnt in contours:
        d=np.cross(p2-p1,cnt-p1)/np.linalg.norm(p2-p1)
        cont_dists.append(abs(np.mean(d)))
    selected_cont_idx= np.argmin(cont_dists)
    
    final_cont = [contours[selected_cont_idx]]
    
    return final_cont


def offset_padline(coordinates, mask, cut_img, distance = 7):
    """
    Offset the whiskerpad contour to the upper left

    Parameters
    ----------
    coordinates : contour coordinates
    mask : mask obtained from when we masked the canny filter detected edges
    cut_img : 2D array from which we detected contours
    distance : Value by which to offset the whisker pad line. The default is 7.

    Returns
    -------
    points : offset contour coordinates

    """
    
    if isinstance(mask, np.bool_):
        mask = np.full((np.shape(cut_img)[0], np.shape(cut_img)[1]), False)
            
    points = [(x1-distance, y1-distance) for x1, y1 in coordinates if mask[int(y1-distance),int(x1-distance)] == False]
        
    return points


def designate_roi_line(p_cont, cut_img, poly_img):
    """
    Designate a region-of-interest line going through our contour coordinates

    Parameters
    ----------
    p_cont : offset contour coordinates
    cut_img : 2D array from which we detected contours
    poly_img : bool 2D array obtained from roi_line_trimmer()

    Returns
    -------
    Xs : x coordinates of roi line
    Ys : y coordinates of roi line

    """

    x = [x[0] for x in p_cont]
    y = [x[1] for x in p_cont]
    mymodel = np.poly1d(np.polyfit(x, y, 2))

    ROIline_xs = np.linspace(0, int(np.shape(cut_img)[1])-1, int(np.shape(cut_img)[1]))
    ROIline_ys = mymodel(ROIline_xs)
    
    ROIline_ys[ROIline_ys > np.shape(cut_img)[0]] = np.nan      
    ROIline_xs[:misc_tools.first_non_nan(ROIline_ys)] = np.nan    
    cleanedXs = [x for x in ROIline_xs if str(x) != 'nan']
    cleanedYs = [y for y in ROIline_ys if str(y) != 'nan']
    
    for i, (x, y) in enumerate(zip(cleanedXs, cleanedYs)):
        if poly_img[int(y),int(x)] == False:
            cleanedXs[i] = np.nan
            cleanedYs[i] = np.nan
    
    Xs = [x for x in cleanedXs if str(x) != 'nan']
    Ys = [y for y in cleanedYs if str(y) != 'nan']
    Xs = Xs[2:-2]
    Ys = Ys[2:-2]
    
    return Xs, Ys


def roi_line_trimmer(cut_img, p1, p2, cont):
    """
    To trim the roi line so it doesn't go into the mouse's body
    Done by drawing a polygon that approximates the 'white' areas of the image

    Parameters
    ----------
    cut_img : 2D array from which we detected contours
    p1 : new xy coord of top left image corner
    p2 : new xy coord of top right image corner
    cont : whiskerpad contour coordinates

    Returns
    -------
    poly_img : TYPE
        DESCRIPTION.

    """
    
    pad_val = 8
    
    b_idx = [i for i, pixel in enumerate(cut_img[-1]) if pixel > 0][1]
    cmin_idx = np.argmin(cont[:,0])
    cmax_idx = np.argmax(cont[:,0])
    
    polygon = [[p1[0]-pad_val, p1[1]-pad_val],
               [p2[0]+pad_val, p2[1]-pad_val],
               [cont[cmax_idx,0]+pad_val-3, cont[cmax_idx,1]-pad_val],
               [cont[cmin_idx,0]-pad_val, cont[cmin_idx,1]+pad_val],
               [b_idx, len(cut_img) - 1]]   
    
    if polygon[2][1] < polygon[1][1]:   
        polygon.pop(1)

    polygon = [np.array(polygon, dtype=np.int32)]
    
    hull_list = []
    for i in range(len(polygon)):
        hull = cv2.convexHull(polygon[i])
        hull_list.append(hull)
        
    poly_img = cv2.fillPoly(cut_img.copy(), hull_list, color=(255,255,255))
    poly_img = cv2.inRange(poly_img, 100, 255).astype(bool)
    
    return poly_img


def extract_raw_whisk(vid_arr, nose_positions, head_directions, crop_size = 100, sample_res = 150, side = 'R', roi_offset = 7, image_edge_offset = 8, plot = True):
    """
    This function reunites all functions in this library (and some from others) into a single function
    to extract the raw whiking pixel-values based on the nose coordinates, the video and head direction coordinates.

    Parameters
    ----------
    vid_arr : A list of 2D arrays (forming a 3D array)
        NR OF FRAMES x VID_Y x VID_X
    nose_positions : pandas dataframe
        contains two columns: 'nose_x' and 'nose_y' with the tracked nose coordinates
    head_directions : list
        defined as the mean slope between front tracker and back tracker.
    crop_size : int, default = 100
        Defines the size of the image cropped around the nose position
    sample_res : int, default = 150
        Defines size to which the ROI line is sampled
    side : str, default = 'R'
        Defines which side of the mouse to extract the raw whisking pixel-values from
    plot : if True, will plot the output using imshow()

    Returns
    -------
    raw_whisking_vals : crop_size*1.5 x len(vid_arr) array
        raw whisker pixel data

    """

    raw_whisking_vals = np.empty((len(vid_arr), sample_res))
    raw_whisking_vals[:] = np.nan

    for index, row in nose_positions.iterrows():

        if not np.isnan(head_directions[index]) and not np.isnan(row['nose_x']):

            # CUT, CROP, ROTATE
            cut_img, p1, p2 = crop_rot_cut(vid_arr[index], row['nose_x'], row['nose_y'], crop_size, head_directions[index], side = side, offset_val = image_edge_offset)

            # DETECT EDGES
            edge_img = detect_edges(cut_img)

            # MASK DETECTED EDGES DUE TO IMAGE ROTATION
            masked_edges, mask = mask_edge_image(cut_img, edge_img, p1, p2)

            # BLUR EDGES
            binary_arr = blur_edges(masked_edges)

            # FIND CONTOURS
            contours = find_edge_segments(binary_arr)

            # SELECT CONTOUR
            contours = select_contour(contours, masked_edges)
            
            if len(contours) > 0:
                
                cont = np.squeeze(contours[0])                
                p_cont = offset_padline(cont, mask, cut_img, distance = roi_offset)
                
                if len(p_cont) > 0:
                    poly_img = roi_line_trimmer(cut_img, p1, p2, cont)

                    try:
                        Xs, Ys = designate_roi_line(p_cont, cut_img, poly_img)
                        if len(Xs) > 0:

                            pixel_vals = np.squeeze(signal.resample(misc_tools.subsample_image(Xs, Ys, cut_img), sample_res))
                            raw_whisking_vals[index] = pixel_vals
                        
                    except IndexError:
                        pass

                        
    if plot:
        plt.figure()
        plt.title('Raw Whisker-Pixel Values')
        plt.imshow(raw_whisking_vals.T)
        plt.xlabel('Time')
        plt.ylabel('Whiskerpad ROI')
        plt.show()
    
    return raw_whisking_vals.T