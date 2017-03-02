import sys
import numpy as np
from generate_submission import generate_submission

things = []
for name in ["02-24-21:38:10.csv", "02-27-19:02:38.csv", "02-28-12:04:43.csv"]:
    arr = np.loadtxt(open("mean/" + name, "rb"), delimiter=",", skiprows=1)[:, 1]
    things.append(arr)

generate_submission(np.mean(things, axis=0))