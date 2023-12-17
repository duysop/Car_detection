from utils.descriptor import hog_features_full_v3
import argparse
import pickle
from termcolor import colored
import os

description = """
feature.py
Usage: python feature.py 
"""
parser = argparse.ArgumentParser(description=description)
parser.add_argument("--window_x", type=int, default=104)
parser.add_argument("--window_y", type=int, default=56)
parser.add_argument("--orientations", type=int, default=9)
parser.add_argument("--pixels_per_cell", type=int, default=8)
parser.add_argument("--cells_per_block", type=int, default=2)
parser.add_argument('--signed_gradient', default=False, type=bool)
parser.add_argument('--gamma_correction', default=False, type=bool)
parser.add_argument('--hnm', default=False, type=bool)
args = parser.parse_args()

orientations: int = args.orientations
pixels_per_cell = (args.pixels_per_cell, args.pixels_per_cell)
cells_per_block = (args.cells_per_block, args.cells_per_block)
signed_gradient = args.signed_gradient
gamma_correction = args.gamma_correction
window_size = (args.window_x, args.window_y)
is_hnm = args.hnm

print(colored("Getting HOG Features: ", "yellow"))
X, y = hog_features_full_v3(orientations = orientations, 
    pixels_per_cell=pixels_per_cell, window_size=window_size,
    cells_per_block=cells_per_block, 
    signed_gradient=signed_gradient,
    gammaCorrection=gamma_correction)
print(colored("Features extracted!", "green"))

if not os.path.exists("data"):
    os.mkdir("data")

file_name = f"data/data_w({window_size[0]},{window_size[1]})_or{orientations}_ppc{pixels_per_cell}_cpb{cells_per_block}_sg{signed_gradient}_gm{gamma_correction}_hnm{is_hnm}.pkl"
config = {
    "window_size": window_size,
    "orientations": orientations,
    "pixels_per_cell": pixels_per_cell,
    "cells_per_block": cells_per_block,
    "signed_gradient": signed_gradient,
    "gamma_correction": gamma_correction,
    "is_hnm": is_hnm
}
file = open(file_name, "wb+")
pickle.dump((X, y, config), file)
file.close()