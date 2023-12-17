import cv2
import argparse
from tqdm import tqdm
import pickle
import os
from utils.detector import sliding_window
from skimage.transform import pyramid_gaussian
from utils.data import iou
import numpy as np
from skimage.feature import hog
from math import ceil

dirname = 'non_cars/'
if not os.path.exists('non_cars'):
    os.mkdir(dirname)

def crop(img, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return img[ y1 : y2, x1: x2,:]

description = """
noncar_img.py
Usage: python hnm.py --window_x 104 --window y 56
"""
parser = argparse.ArgumentParser(description=description)
parser.add_argument("--downscale", type=float, default= 1.5)
parser.add_argument("--iou_threshold", type=float, default=0.0)
parser.add_argument("--session", type=str, default="data_w(104,56)_or9_ppc(8, 8)_cpb(2, 2)_sgFalse_gmFalse_hnmFalse")
parser.add_argument("--examples", type=int, default=1e10)
args = parser.parse_args()

session = args.session[4:]
print(f"Hard Negative Mining for {args.session}")
file = open(f"data/data{session}.pkl", "rb")
_, _, config = pickle.load(file)
file.close()
assert config['is_hnm'] == False

window_size = config['window_size']
n_examples = args.examples

orientations: int = config['orientations']
pixels_per_cell = config['pixels_per_cell']
cells_per_block = config['cells_per_block']
downscale = args.downscale
iou_threshold = args.iou_threshold

step_size = (ceil(window_size[0] / 2), ceil(window_size[1] / 2))
signed_gradient = config['signed_gradient']
gamma_correction = config['gamma_correction']

blockSize = (cells_per_block[1] * pixels_per_cell[1], cells_per_block[0] * pixels_per_cell[0])
blockStride = (pixels_per_cell[1], pixels_per_cell[0])

session = f"_w({window_size[0]},{window_size[1]})_or{orientations}_ppc{pixels_per_cell}_cpb{cells_per_block}_sg{signed_gradient}_gm{gamma_correction}_hnmFalse"

file = open(f"data/svm{session}.pkl", "rb")
clf = pickle.load(file)
file.close()

img_path = "raw/image_kitti/"
label_path = "raw/label_kitti/"

ignore = ["Pedestrian", "Person_sitting", "Cyclist", "Tram"]

label_dict = dict()

def to_float_array(array):
    return [float(x) for x in array]

for img_label_name in tqdm(os.listdir(label_path)):
    index = img_label_name.split(".")[0]
    img_label_path = label_path + img_label_name
    with open(img_label_path, "r") as file:
        data = [line.strip().split(" ") for line in file.readlines()] 
        data = [ (entry[0], to_float_array(entry[4:8])) for entry in data if entry[0] not in ignore]
        label_dict[index] = data

def isCarTypes(box, img_name):
    img_index = img_name.split(".")[0]
    for entry in label_dict[img_index]:
        if iou(entry[1], box) > iou_threshold:
            return True
    return False

def rgb2gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return np.uint8(gray)

count = 0

for img_name in tqdm(os.listdir(img_path)):
    index = img_name.split(".")[0]
    img = cv2.imread(img_path + img_name, cv2.IMREAD_UNCHANGED)

    scale = 0
    for j, im_scaled in enumerate(pyramid_gaussian(img, downscale = downscale, preserve_range=True, channel_axis=2, max_layer=2)):
        
        if im_scaled.shape[0] < window_size[1] or im_scaled.shape[1] < window_size[0]:
            break

        for i, (x, y, im_window) in enumerate(sliding_window(im_scaled, window_size, step_size)):
            if im_window.shape[0] != window_size[1] or im_window.shape[1] != window_size[0]:
                continue
            x1 = int(x * (downscale**scale))
            y1 = int(y * (downscale**scale))
            x2 = x1 + int(window_size[0] * (downscale**scale))
            y2 = y1 + int(window_size[1] * (downscale**scale))
            box = [x1, y1, x2, y2] 

            if isCarTypes(box, img_name) == False:
                if count > n_examples:
                    print(f"Reach maximum number of examples of {n_examples}")
                    exit(0)

                gray_window = rgb2gray(im_window)
                
                hog_descriptor = cv2.HOGDescriptor(
                    _winSize=window_size,
                    _blockSize=blockSize,
                    _blockStride=blockStride,
                    _signedGradient=signed_gradient,
                    _cellSize=pixels_per_cell,
                    _nbins=orientations,
                    _gammaCorrection=gamma_correction)
                
                fd = hog_descriptor.compute(gray_window).flatten()

                fd = fd.reshape(1, -1)
                pred = clf.predict(fd)

                if pred == 1:
                    cv2.imwrite(dirname + index + "_hard" + str(j) + "_" + str(i) + '.jpg', im_window)
                    count += 1
                    
        scale += 1    

print(f"Number of hard examples: {count}")

with open(f"data/log{session}.txt", "a+") as file:
    print(f"Number of hard examples: {count}", file = file) 
