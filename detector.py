import numpy as np 
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
# from skimage.feature import hog
import cv2
import matplotlib.pyplot as plt 
from utils.detector import sliding_window
import pickle
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

def rgb2gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return np.uint8(gray)

def _detect(image, clf, scale, downscale, threshold, config):

    min_wdw_sz = config['window_size'] if "window_size" in config.keys() else (104, 56)
    orientations: int = config['orientations'] if "orientations" in config.keys() else 18
    pixels_per_cell = config['pixels_per_cell'] if "pixels_per_cell" in config.keys() else (8, 8)
    cells_per_block = config['cells_per_block'] if "cells_per_block" in config.keys() else (2, 2)
    signed_gradient = config['signed_gradient'] if "signed_gradient" in config.keys() else False
    gamma_correction = config['gamma_correction'] if "gamma_correction" in config.keys() else False
    blockSize = (cells_per_block[1] * pixels_per_cell[1], cells_per_block[0] * pixels_per_cell[0])
    blockStride = (pixels_per_cell[1], pixels_per_cell[0])

    hog = cv2.HOGDescriptor(
                        _winSize=min_wdw_sz,
                        _blockSize=blockSize,
                        _blockStride=blockStride,
                        _signedGradient=signed_gradient,
                        _gammaCorrection=gamma_correction,
                        _cellSize=pixels_per_cell,
                        _nbins=orientations)
    step_size = (5, 5)
    
    #The list contains detections at the current scale
    detections = []

    for (x, y, im_window) in sliding_window(image, min_wdw_sz, step_size):

            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue

            gray_window = rgb2gray(im_window)

            fd = hog.compute(gray_window).flatten()

            fd = fd.reshape(1, -1)
            pred = clf.predict(fd)

            if pred == 1:
                if clf.decision_function(fd) > threshold:
                    detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), clf.decision_function(fd), 
                    int(min_wdw_sz[0] * (downscale**scale)),
                    int(min_wdw_sz[1] * (downscale**scale))))

    return detections


def detector(filename, clf, config: dict):
    im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    downscale = 1.25
    threshold = 0
    min_wdw_sz = config['window_size'] if "window_size" in config.keys() else (104, 56)
    print(config)
    
    executor = ProcessPoolExecutor(max_workers=5)
    futures = []
    scale = 0
    start = time.time()

    for im_scaled in pyramid_gaussian(im, max_layer=2, downscale = downscale, preserve_range=True, channel_axis=2):

        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break

        futures.append(executor.submit(_detect, im_scaled, clf, scale, downscale, threshold, config)) 
        
        scale += 1
    
    detections = []
    for future in as_completed(futures):
        _detections = future.result()
        detections.extend(_detections)

    # Kích thước của lưới ô vuông.
    clone = im.copy()

    for (x_tl, y_tl, _, w, h) in detections:
        cv2.rectangle(im, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness = 2)
    
    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])

    sc = [score[0] for (x, y, score, w, h) in detections]

    for(xA, yA, xB, yB) in rects:
        cv2.rectangle(im, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
    print("sc: ", sc)

    pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.25)
    print("shape, ", pick.shape)

    for(xA, yA, xB, yB) in pick:
        cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)

    end = time.time()
    print(f"Execution time: {end - start} s / frame")
    plt.axis("off")
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.title("Raw Detection before NMS")
    plt.show()

    plt.axis("off")
    plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
    plt.title("Final Detections after applying NMS")
    plt.show()


if __name__ == "__main__":

    description = """
    detector.py
    Usage: python detector.py --image testcase.png
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--image", type=str, default="raw/test_case/test_case1.jpg")
    parser.add_argument("--session", type=str, default="data_w(104,56)_or9_ppc(8, 8)_cpb(2, 2)_sgFalse_gmFalse_hnmTrue")
    args = parser.parse_args()

    session = args.session[4:]
    
    file = open(f"data/svm{session}.pkl", "rb")
    clf = pickle.load(file)
    file.close()

    file = open(f"data/data{session}.pkl", "rb")
    _, _, config = pickle.load(file)
    file.close()

    image_name = args.image
    detector(image_name, clf, config)