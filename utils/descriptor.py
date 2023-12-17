import cv2 
import numpy as np
import os
from tqdm import tqdm

def rgb2gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return np.uint8(gray)

def hog_features_full_v3(
    cars_root = "cars/", 
    non_cars_root = "non_cars/",
    orientations: int = 9, 
    window_size = (104, 56),
    signed_gradient: bool = False,
    pixels_per_cell = (8, 8), 
    cells_per_block = (2, 2),
    gammaCorrection: bool = False
    ) -> np.ndarray:

    images_features = []
    cars_paths = os.listdir(cars_root)
    cars_paths = [cars_root + path for path in  cars_paths]
    non_cars_paths = os.listdir(non_cars_root)
    non_cars_paths = [non_cars_root + path for path in  non_cars_paths]

    blockSize = (cells_per_block[1] * pixels_per_cell[1], cells_per_block[0] * pixels_per_cell[0])
    blockStride = (pixels_per_cell[1], pixels_per_cell[0])

    hog = cv2.HOGDescriptor(
                        _winSize=window_size,
                        _blockSize=blockSize,
                        _blockStride=blockStride,
                        _signedGradient=signed_gradient,
                        _cellSize=pixels_per_cell,
                        _nbins=orientations,
                        _gammaCorrection=gammaCorrection)

    for img_path in tqdm(cars_paths):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = rgb2gray(img)

        features = hog.compute(img).flatten()

        images_features.append(features)

    for img_path in tqdm(non_cars_paths):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = rgb2gray(img)

        features = hog.compute(img).flatten()
        
        images_features.append(features)

    n_cars = len(cars_paths)
    n_non_cars = len(non_cars_paths)
    labels = [1] * n_cars + [0] * n_non_cars
    return images_features, labels