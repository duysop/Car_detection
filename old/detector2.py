import numpy as np 
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
from skimage.feature import hog
import cv2
import matplotlib.pyplot as plt 
from utils.detector import sliding_window_vectorized
import pickle
import argparse
import time


def detector(filename, clf, config: dict):
    im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    min_wdw_sz = config['window_size'] if "window_size" in config.keys() else (104, 56)
    downscale = 1.05
    threshold = 1
    print(config)

    detections = []
    scale = 0
    start = time.time()

    for im_scaled in pyramid_gaussian(im, downscale = downscale, preserve_range=True):

        im_scaled = np.uint8(im_scaled)

        #The list contains detections at the current scale
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break

        locs, feats = sliding_window_vectorized(im_scaled, config)

        # pred = clf.predict(feats)
        scores = clf.decision_function(feats)

        where1 = np.where(scores > threshold)[0]

        for index in where1:
            # if scores[index] > threshold:
                detections.append(
                    (int(locs[index][0] * (downscale**scale)), 
                    int(locs[index][1] * (downscale**scale)), 
                    scores[index], 
                    int(min_wdw_sz[0] * (downscale**scale)),
                    int(min_wdw_sz[1] * (downscale**scale)))
                    )
        
        scale += 1

    # Kích thước của lưới ô vuông.
    clone = im.copy()

    for (x_tl, y_tl, _, w, h) in detections:
        cv2.rectangle(im, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness = 2)
    
    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])

    sc = [score for (x, y, score, w, h) in detections]

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
    parser.add_argument("--image", type=str, default="raw/test_case/test_case4.jpg")
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