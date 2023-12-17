from scipy.io.matlab import loadmat
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm
import os
dirname = 'cars/'
if not os.path.exists('cars'):
    os.mkdir(dirname)

description = """
car_img.py
Usage: python car_img.py --window_x 104 --window y 56
"""
parser = argparse.ArgumentParser(description=description)
parser.add_argument("--window_x", type=int, default=104)
parser.add_argument("--window_y", type=int, default=56)
args = parser.parse_args()

window_size = (args.window_x, args.window_y)

train_annos_path = "raw/car_devkit/devkit/cars_train_annos.mat"
train_annos = loadmat(train_annos_path)

test_annos_path = "raw/car_devkit/devkit/cars_test_annos.mat"
test_annos = loadmat(test_annos_path)

def crop(img, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return img[ y1 : y2, x1: x2,:]

for image_stat in tqdm(train_annos['annotations'][0]):
    # print(image_stat)
    x1, y1 = image_stat[0].item(), image_stat[1].item()
    x2, y2 = image_stat[2].item(), image_stat[3].item()
    img_class = image_stat[4].item()
    img_name = image_stat[5].item()
    # print((x1, y1), (x2, y2))

    img = plt.imread(f'raw/cars_train/{img_name}', cv2.IMREAD_UNCHANGED)
    if len(img.shape) != 3:
        # print(f"Image in gray scale: {img.shape}")
        continue

    croped_img = crop(img, (x1, y1), (x2, y2))
    resized = cv2.resize(croped_img, window_size, interpolation = cv2.INTER_AREA)
    cv2.imwrite(dirname + "train_" + img_name, resized)

for image_stat in tqdm(test_annos['annotations'][0]):
    # print(image_stat )
    x1, y1 = image_stat[0].item(), image_stat[1].item()
    x2, y2 = image_stat[2].item(), image_stat[3].item()
    img_name = image_stat[4].item()
    # print((x1, y1), (x2, y2))

    img = plt.imread(f'raw/cars_test/{img_name}', cv2.IMREAD_UNCHANGED)
    if len(img.shape) != 3:
        # print(f"Image in gray scale: {img.shape}")
        continue

    croped_img = crop(img, (x1, y1), (x2, y2))
    resized = cv2.resize(croped_img, window_size, interpolation = cv2.INTER_AREA)
    cv2.imwrite(dirname + "test_" + img_name, resized)


# img_path = "raw/image_kitti/"
# label_path = "raw/label_kitti/"

# ignore = ["Pedestrian", "Person_sitting", "Cyclist", "Tram", "DontCare"]

# from math import ceil
# label_dict = dict()
# def to_int_array(array):
#     return [ceil(float(x)) for x in array]
# for img_label_name in tqdm(os.listdir(label_path)):
#     index = img_label_name.split(".")[0]
#     img_label_path = label_path + img_label_name
#     with open(img_label_path, "r") as file:
#         data = [line.strip().split(" ") for line in file.readlines()] 
#         data = [ (entry[0], to_int_array(entry[4:8])) for entry in data if entry[0] not in ignore]
#         label_dict[index] = data
    
# for img_name in tqdm(os.listdir(img_path)):
#     img = cv2.imread(img_path + img_name, cv2.IMREAD_UNCHANGED)
#     img_index = img_name.split(".")[0]
#     for i, entry in enumerate(label_dict[img_index]):
#         box = entry[1]
#         croped_img = crop(img, (box[0], box[1]), (box[2], box[3]))
#         if croped_img.shape[1] < window_size[0] or croped_img.shape[0] < window_size[1]:
#             continue
#         resized = cv2.resize(croped_img, window_size, interpolation = cv2.INTER_AREA)
#         cv2.imwrite(dirname + img_index + "_" + str(i) + '.jpg', resized)