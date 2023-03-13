#!/usr/bin/env python
# coding: utf-8

# In[4]:


import json
import os
import openslide
from PIL import Image, ImageDraw
import cv2
import numpy as np


# In[5]:


def split_rectangle_into_squares(left_upper_coord, lower_right_coord, size, drop_last):
    """Splits rectangle into squares with the given size. If drop_last == 1, squares can be only size*size"""
    x1, y1 = left_upper_coord
    x2, y2 = lower_right_coord
    squares = []
    if (drop_last == 0):
        for x in range(x1, x2, size):
            for y in range(y1, y2, size):
                    squares.append(((x, y), (min(x2, x+size), min(y2, y+size))))
    
    elif (drop_last == 1):
        for x in range(x1, x2, size):
            for y in range(y1, y2, size):
                if (x + size <= x2 and y + size <= y2):
                    squares.append(((x, y), (x+size, y+size)))
  
    return squares


def check_if_the_image_is_background (image):
    """if this value is more than a predetermined threshold (75% of the image area), then it is mostly blank space background"""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]

    total_pixels = thresh.shape[0] * thresh.shape[1]
    white_pixels = cv2.countNonZero(thresh)
    ratio = white_pixels / total_pixels * 100
    
    #Mostly Background
    if ratio >= 75:
        return True
    
    #Not Mostly Background
    else:
        return False


def empty_folder(folder):
    """Makes given folder empty"""
    import os, shutil
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def cut_slide_in_tiles (svs, name_of_the_slide, path_to_save, left_upper_coord, lower_right_coord, size_in_mm = 200, drop_last = True, preserve = False, check_background = False):
    """
    This function returns list of tiles size*size (if drop_last == 1)

    Parameters
    ----------
    svs: str
        path to directory where .svs slides are located
    
    name_of_the_slide: str
        name of the svs slide, that we need to cut 
    
    path_to_save: str
        path to directory where we want to save our cutted tiles    
        
    left_upper_coord: (x, y)
        left upper coordinate of the rectangle, that we need to cut 
        
    lower_right_coord: (x, y)
        lower right coordinate of the rectangle, that we need to cut
    
    size_in_mm: size in micrometres
        size of the needed tiles 
    
    drop_last: bool, optional
        default: True
        if True then we drop tiles that are not size*size
        of False then we include tiles that are not size*size
        
    preserve: bool, optional
        default: false
        whether to preserve target directory contents
        
    check_background: bool, optional
        default: false
        if we want to check if the image is mostly background and if it is, don't save it 
    
    result: 
        saves, cutted with the given parametrs tiles, to the given directory
    """
    #size in pixels:
    size = int(size_in_mm / 0.25316455696)
    if not os.path.isdir(svs):
        raise BaseException("No such directory: " + svs)
    
    if not os.path.isdir(path_to_save):
        print("Creating directory " + path_to_save + "...")
        os.mkdir(path_to_save)
    if not os.path.isdir(os.path.join(path_to_save, name_of_the_slide)):
        print("Creating directory " + name_of_the_slide + "...")
        os.mkdir(os.path.join(path_to_save, name_of_the_slide))

    else:
        print('Destination folder is not empty')
        if not preserve:
            print('Removing all files...')
            empty_folder(path_to_save)
            
    
    slide = openslide.open_slide(os.path.join(svs, name_of_the_slide + '.svs'))
    
    list_of_squares_coord = split_rectangle_into_squares(left_upper_coord, lower_right_coord, size, drop_last)
    if (len(list_of_squares_coord) == 0):
        raise BaseException("No squares detected")
    
    tile_number = 0
    for square in list_of_squares_coord:
        x_upper_left, y_upper_left = square[0][0], square[0][1]
        x_lower_right, y_lower_right = square[1][0], square[1][1]
        region = slide.read_region((x_upper_left, y_upper_left), 0, (x_lower_right - x_upper_left, y_lower_right - y_upper_left))
        #save region to the directory
        if not (check_background):
            region.save(os.path.join(path_to_save,
                            name_of_the_slide, name_of_the_slide + "_" + str(tile_number) + '.png'), "PNG")
            
        elif (check_background): 
            if not (check_if_the_image_is_background(region)):
                region.save(os.path.join(path_to_save,
                            name_of_the_slide, name_of_the_slide + "_" + str(tile_number) + '.png'), "PNG")
                
        tile_number += 1
        
    return None


# In[6]:


slide_layout_path = "/home/alexmak123/slide_layout"
#NOTE: in PATH_TO_SAVE directory all the data will be deleted
PATH_TO_SAVE = "/home/alexmak123/result_tiles_for_validation"

slides = os.listdir(slide_layout_path)

#NOTE: max size of the slide ∽ (0,0)x(100 000, 100 000)
for slide_to_check in slides:
    svs = os.path.join(slide_layout_path, slide_to_check)
    cut_slide_in_tiles(svs, slide_to_check, PATH_TO_SAVE, (0,0), (50000,50000))


# In[ ]:




