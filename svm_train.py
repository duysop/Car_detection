from sklearn.svm import LinearSVC
import argparse
import pickle
from termcolor import colored
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

description = """
svm.py
Usage: python svm.py 
"""
parser = argparse.ArgumentParser(description=description)
parser.add_argument("--session", type=str, default="data_w(104,56)_or9_ppc(8, 8)_cpb(2, 2)_sgFalse_gmFalse_hnmFalse")
args = parser.parse_args()

session = args.session[4:]
assert len(session) > 20

file = open(f"data/data{session}.pkl", "rb")
X, y, _ = pickle.load(file)
file.close()
print(colored("Start training: ", "yellow"))
start = time.time()
clf = LinearSVC(max_iter= 3000)
clf.fit(X, y)
end = time.time()
print(colored(f"Train SVM completed! Time: {end - start} s", "green"))
  
file = open(f"data/svm{session}.pkl", "wb+")
pickle.dump(clf, file)
file.close()


