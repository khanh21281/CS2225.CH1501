# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 23:15:19 2020

@author: Admin
"""


# Step 1 — Image transformations
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
import os

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

# Step 2 — List all the files in a folder and read them
# our folder path containing some images
folder_path = 'D:\ML\dat'
# the number of file to generate
num_files_desired = 900

# loop on all files of the folder and build a list of files paths
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

num_generated_files = 120
while num_generated_files <= num_files_desired:
    # random image from the folder
    image_path = random.choice(images)
    # read image as an two dimensional array of pixels
    image_to_transform = io.imread(image_path)
    

# Step 3 — Images transformation
# dictionary of the transformations functions we defined earlier
    available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip
    }

    # random num of transformations to apply
    num_transformations_to_apply = random.randint(1, len(available_transformations))

    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        # choose a random transformation to apply for a single image
        key = random.choice(list(available_transformations))
        transformed_image = available_transformations[key](image_to_transform)
        num_transformations += 1


# Step 4 — Save the new images
# define a name for our new file
        new_file_path = 'D:\\ML\\Data_khanh\\user.2.%s.jpg' %(num_generated_files)

# write image to the disk
        io.imsave(new_file_path, transformed_image)
    num_generated_files += 1