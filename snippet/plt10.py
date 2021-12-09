import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

fig, ax = plt.subplots(figsize=(8, 6))
rc("text", usetex=True)
rc("font", size=16)

with open("snippet/data/N12L001K1/analyzer_result.txt") as f:
    reader = csv.reader(f)
    L001K1 = [float(row[0]) for row in reader]
    L001K1_hist, bin_edges = np.histogram(L001K1, range=(0, 15), bins=15 * 4)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

with open("snippet/data/N12L001K10/analyzer_result.txt") as f:
    reader = csv.reader(f)
    L001K10 = [float(row[0]) for row in reader]
    L001K10_hist, _ = np.histogram(L001K10, range=(0, 15), bins=15 * 4)


with open("snippet/data/N12L001K50/analyzer_result.txt") as f:
    reader = csv.reader(f)
    L001K50 = [float(row[0]) for row in reader]
    L001K50_hist, _ = np.histogram(L001K50, range=(0, 15), bins=15 * 4)

with open("snippet/data/N12L001K100/analyzer_result.txt") as f:
    reader = csv.reader(f)
    L001K100 = [float(row[0]) for row in reader]
    L001K100_hist, _ = np.histogram(L001K100, range=(0, 15), bins=15 * 4)

with open("snippet/data/N12L01K10/analyzer_result.txt") as f:
    reader = csv.reader(f)
    L01K10 = [float(row[0]) for row in reader]
    L01K10_hist, _ = np.histogram(L01K10, range=(0, 15), bins=15 * 4)


plt.plot(bin_centers, L001K1_hist, label="K\(\pm\)1\% L\(\pm\)0.01\%")
plt.plot(bin_centers, L001K10_hist, label="K\(\pm\)10\% L\(\pm\)0.01\%")
plt.plot(bin_centers, L001K50_hist, label="K\(\pm\)50\% L\(\pm\)0.01\%")
plt.plot(bin_centers, L001K100_hist, label="K\(\pm\)100\% L\(\pm\)0.01\%")
plt.plot(bin_centers, L01K10_hist, label="K\(\pm\)10\% L\(\pm\)0.1\%")
plt.ylabel("frequency", size=20)
plt.xlabel("evaluation function value", size=20)
plt.legend(loc="best")
plt.show()
