import numpy as np 
import cv2
def sliding_window(image: np.ndarray, window_size = (104, 56), step_size = (16, 16)):
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])


def sliding_window_vectorized(image: np.ndarray, config: dict):

    cell_size = config['pixels_per_cell'] if "pixels_per_cell" in config.keys() else (8, 8)
    block_size = config['cells_per_block'] if "cells_per_block" in config.keys() else (2, 2)
    nbins = config['orientations'] if "orientations" in config.keys() else 18
    winSize = config['window_size'] if "window_size" in config.keys() else (104, 56)
    blockSize = (block_size[1] * cell_size[1], block_size[0] * cell_size[0])
    blockStride = (cell_size[1], cell_size[0])
    signed_gradient = config['signed_gradient'] if "signed_gradient" in config.keys() else False
    gamma_correction = config['gamma_correction'] if "gamma_correction" in config.keys() else False

    hog = cv2.HOGDescriptor(
                            _winSize=winSize,
                            _blockSize=blockSize,
                            _blockStride=blockStride,
                            _signedGradient=signed_gradient,
                            _gammaCorrection=gamma_correction,
                            _cellSize=cell_size,
                            _nbins=nbins)
    # Kích thước của lưới ô vuông.
    n_cells = (winSize[0] // cell_size[0], winSize[1] // cell_size[1])

    hog_feats = hog.compute(image)\
               .reshape(-1, n_cells[1] - block_size[1] + 1,
                        n_cells[0] - block_size[0] + 1,
                        block_size[0], block_size[1], nbins) \
                .transpose((0, 2, 1, 3, 4, 5))

    y = (image.shape[0] - winSize[1]) // blockStride[1] + 1
    x = (image.shape[1] - winSize[0]) // blockStride[0] + 1
    assert hog_feats.shape[0] == x * y, f"{hog_feats.shape[0]} != {x * y}"

    x_locs = np.arange(0, image.shape[1] - winSize[0] + 1, blockStride[0])
    y_locs = np.arange(0, image.shape[0] - winSize[1] + 1, blockStride[1]) 

    y_locs = np.repeat(y_locs, len(x_locs))
    x_locs = np.tile(x_locs, len(y_locs))
    indexes = list(zip(x_locs, y_locs))
    feats = hog_feats.reshape(x * y, -1)
    assert len(feats) == len(indexes)
    
    return indexes, feats