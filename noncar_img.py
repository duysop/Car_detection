import cv2
import argparse
from tqdm import tqdm
import os
from utils.detector import sliding_window
from skimage.transform import pyramid_gaussian
from utils.data import iou
import numpy as np
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
Usage: python noncar_img.py --window_x 104 --window y 56
"""
parser = argparse.ArgumentParser(description=description)
parser.add_argument("--window_x", type=int, default=104)
parser.add_argument("--window_y", type=int, default=56)
parser.add_argument("--examples", type=int, default=1e10)
parser.add_argument("--downscale", type=float, default= 1.5)
parser.add_argument("--iou_threshold", type=float, default=0.0)
args = parser.parse_args()

window_size = (args.window_x, args.window_y)
step_size = (ceil(args.window_x / 2), ceil(args.window_y / 2))
n_examples = args.examples
downscale = args.downscale
iou_threshold = args.iou_threshold

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


count = 0

images_path = os.listdir(img_path)
np.random.shuffle(images_path)

for img_name in tqdm(images_path):
    index = img_name.split(".")[0]
    img = cv2.imread(img_path + img_name, cv2.IMREAD_UNCHANGED)
    scale = 0
    for j, im_scaled in enumerate(pyramid_gaussian(img, downscale = downscale, channel_axis=2, max_layer=2)):

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
                cv2.imwrite(dirname + index + "_" + str(j) + "_" + str(i) + '.jpg', im_window * 255)
                count += 1
                
        scale += 1
