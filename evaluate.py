###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012                   #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012                   #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: May 24th 2018                                                 #
###########################################################################################

from metrics import BoundingBox
from metrics import BoundingBoxes
from metrics.Evaluator import *
from metrics.utils import *
import pickle

# TODO
session = r"data_w(96,48)_or18_ppc(8, 8)_cpb(2, 2)_sgTrue_gmFalse_hnmTrue"

groundtruth_path = "data/"

currentPath = f"../../data/evaluation/{session}"

def getBoundingBoxes():
    """Read txt files containing bounding boxes (ground truth and detections)."""
    allBoundingBoxes = BoundingBoxes()
    import glob
    import os
    # Read ground truths
    global currentPath
    global groundtruth_path
    folderGT = os.path.join(groundtruth_path, 'ground_truths')
    os.chdir(folderGT)
    files = glob.glob("*.txt")
    files.sort()
    # Class representing bounding boxes (ground truths and detections)
    allBoundingBoxes = BoundingBoxes()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            x = float(splitLine[1])  # confidence
            y = float(splitLine[2])
            w = float(splitLine[3])
            h = float(splitLine[4])
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                CoordinatesType.Absolute, (200, 200),
                BBType.GroundTruth,
                format=BBFormat.XYWH)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    # Read detections
    folderDet = os.path.join(currentPath, 'detection_bounding_boxes').replace("\\", "/")
    os.chdir(folderDet)
    files = glob.glob("*.txt")
    files.sort()
    # Read detections from txt file
    # Each line of the files in the detections folder represents a detected bounding box.
    # Each value of each line is  "class_id, confidence, x, y, width, height" respectively
    # Class_id represents the class of the detected bounding box
    # Confidence represents confidence (from 0 to 1) that this detection belongs to the class_id.
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        # nameOfImage = f.replace("_det.txt","")
        nameOfImage = f.replace(".txt", "")
        # Read detections from txt file
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            confidence = float(splitLine[1])  # confidence
            x = float(splitLine[2])
            y = float(splitLine[3])
            w = float(splitLine[4])
            h = float(splitLine[5])
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                CoordinatesType.Absolute, (200, 200),
                BBType.Detected,
                confidence,
                format=BBFormat.XYWH)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    return allBoundingBoxes

# Read txt files containing bounding boxes (ground truth and detections)
boundingboxes = getBoundingBoxes()
# Create an evaluator object in order to obtain the metrics
evaluator = Evaluator()
##############################################################
# VOC PASCAL Metrics
##############################################################
# Get metrics with PASCAL VOC metrics
metricsPerClass = evaluator.GetPascalVOCMetrics(
    boundingboxes,  # Object containing all bounding boxes (ground truths and detections)
    IOUThreshold=0.5,  # IOU threshold
    method=MethodAveragePrecision.EveryPointInterpolation)  # As the official matlab code
print("Average precision values per class:\n")
# Loop through classes to obtain their metrics
for mc in metricsPerClass:
    # Get metric values per each class
    c = mc['class']
    precision = mc['precision']
    recall = mc['recall']
    average_precision = mc['AP']
    ipre = mc['interpolated precision']
    irec = mc['interpolated recall']
    # Print AP per class
    print('%s: %f' % (c, average_precision))
    print(precision)
    print(recall)
os.chdir("..")
file = open(f"result.txt", "w+")
print('%s: %f' % (c, average_precision), file=file)
print('precision: ', str(precision), file=file)
print('recall: ', str(recall), file=file)
file.close()

file = open(f"result.pkl", "wb+")
pickle.dump(mc, file=file)
file.close()


