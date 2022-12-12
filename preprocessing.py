# Author: Ellsworth McCullough
# File Name: preprocessing
# Description:
# Retrieves photo data from the directory given in arguments, processes it into data
# usable in regression.

# hpfc imports
from hpfc_standard import *

# Library imports
from sklearn.decomposition import PCA
import os
import cv2
import mediapipe as mp
import numpy as np
import random as ran
import math


# Manager function preprocessing
# Manages calls and returns of every main function in preprocessing
# Returns two HFPC data objects,
# first will be sample data, second will be testing data
def preprocessing(args):

    # Retrieves photonames
    photo_names = photo_name_retrieval(args)

    # Retrieves facial landmarks, heights, and indexes
    cords, height, index = photo_conversion(photo_names, args)

    # Retrieves usable data
    data = feature_reduction(cords, args)

    # Returns sample and testing data
    return data_separation(data, height, index, args)


# Retrieves photonames from the directory given in arguments
# Returns a numpy array of photonames
def photo_name_retrieval(args):

    # retrieves directory name
    directory = args.at[0, 'directory']

    # holds the filenames in the given directory
    photonames = []

    # goes through all files in the given directory
    for photoname in os.listdir(directory):
        # inputs the dir//filename into the filenames
        photonames.append(os.path.join(directory, photoname))

    # returns the retrieved filenames
    return np.asarray(photonames)


# Converts the photos from the directory into usable data
# Returns three arrays
# First is coordinates, second is heights, third is indexes
def photo_conversion(photoname, args):

    # Used to be more easily understood and read
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    # Arrays to data retrieved from mediapipe
    landmarks = []

    # Arrays to hold data retrieved from filenames
    height = []
    index = []

    # Retrieves data using a combination of opencv and mediapipe
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        for i, file in enumerate(photoname):

            # Retrieving height
            # Filenames in format of ?\index_height.type
            filename = file.split("_")

            # Activates if \ exists in path
            if "\\" in filename[0]:
                # Filename_path in format of ?\index
                filename_path = filename[0].split("\\")
                # Takes the index portion of the split string
                index.append(filename_path[int(len(filename_path) - 1)])
            # Activates if \ does not exist in path
            else:
                # Takes the index portion of filename
                index.append(int(filename[0]))

            # Takes the height portion of the split string
            height.append(int(filename[1].split(".")[0]))

            # Retrieving facial landmarks
            image = cv2.imread(file)
            # Convert the BGR image to RGB before retrieving landmarks
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            results = results.multi_face_landmarks

            # Inserts results into an array for later use
            landmarks.append(results)

    # Array to hold arrays containing coordinate values,
    # expected objects are [x, y, z]
    cords = []

    # Goes through all landmarks and retrieves x, y, and z components
    for i, el_i in enumerate(landmarks):
        for j, el_j in enumerate(el_i):

            # Array to hold coordinate values before appending
            # [x, y, z]
            cords_lesser = []

            # Loops though the current landmark retrieving the x, y, z and components
            for k in range(int(args.at[0, 'fla'])):
                cords_lesser.append([el_j.landmark[k].x,
                                     el_j.landmark[k].y,
                                     el_j.landmark[k].z])

            # Appends most recent coordinate array
            cords.append(cords_lesser)


    # Converts cords to numpy array
    cords = np.asarray(cords)
    # Retrieves 3-dimensional components
    cords_ele, cords_x, cords_y = cords.shape
    # Converts cords from 3-dimensional to 2-dimensional
    # Should result in no loss of data
    cords = np.reshape(cords, (cords_ele, cords_x*cords_y))

    return cords, height, index


# Reduces features of given data using principle component analysis
# Reduces to amount of components specified in arguments
def feature_reduction(data, args):

    # Sets the principle components to the minimum between x and y of data, and pc given in args
    pc = min(len(data), len(data[0]), int(args['pc'][0]))
    # Creates a PCA object with the amount of components specified in args
    pca = PCA(n_components = pc)

    # Performs the feature reduction
    data = pca.fit_transform(data)

    return data


# Separates the data into sample and testing data
# Sample ratio dependent on ran argument given in args
# Organizes randomized separation based on height
# from lowest to highest
def data_separation(data, height, index, args):

    # Sample arrays
    data_s = []
    height_s = []
    index_s = []

    # Testing Arrays
    data_t = []
    height_t = []
    index_t = []

    # Uses system time to seed random number generator
    ran.seed()

    # Sets ran to modulo val
    # modulo value used to determine whether to add data row to
    # sample data or testing data, row only added to testing data
    # when chance % modulo_val = 0. Chances for testing data
    # roughly 1/modulo_val
    modulo_val = int(args['ran'])

    # Maximum value to be used in randrange
    chance_max = modulo_val*10

    # Activates if data, height, and index do not have equal array lengths
    if not (len(data) == len(height) == len(index)):
        error_message(data_separation.__name__,
                      'Unequal array lengths')
    # Activates if data, height, and index do have equal array lengths
    else:
        # Goes through every element in data, data, height, and index have equal
        # array elements
        for i in range(len(data)):

            # Generates a random number between 0 and chance_max
            chance = ran.randrange(chance_max)

            # Creates data organized from smallest to largest height
            if bool(args.at[0, 'ordered']):
                # Modulos that value, if not 0 adds row to sample dataset
                if chance % modulo_val != 0:
                    data_s, height_s, index_s = \
                        data_append(data_s, height_s, index_s,
                                data, height, index)
                # Modulos the value, if 0 adds row to testing dataset
                else:
                    data_t, height_t, index_t = \
                        data_append(data_t, height_t, index_t,
                                data, height, index)
            # Creates unorganized data
            else:
                # Modulos that value, if not 0 adds row to sample dataset
                if chance % modulo_val != 0:
                    data_s.append(data[i])
                    height_s.append(height[i])
                    index_s.append(index[i])
                # Modulos the value, if 0 adds row to testing dataset
                else:
                    data_t.append(data[i])
                    height_t.append(height[i])
                    index_t.append(height[i])


    # Creates two HPFCData objects representing sample data and
    # testing data
    dataset_s = HPFCData(index_s, height_s, data_s)
    dataset_t = HPFCData(index_t, height_t, data_t)

    return dataset_s, dataset_t


# Appends to three arrays based on the lowest height value in height_old
# Alters height_old after appending values
# Returns three arrays,
# first representing general data, second height, third index
def data_append(data, height, index, data_old, height_old, index_old):

    # Retrieves index of lowest value
    lowest = height_old.index(min(height_old))

    # Appends old arrays to new arrays based on lowest height val
    data.append(data_old[lowest])
    height.append(height_old[lowest])
    index.append(index_old[lowest])

    # Sets the previous lowest to the highest possible value
    height_old[lowest] = math.inf

    return data, height, index

