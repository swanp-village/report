import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

fig, ax = plt.subplots(figsize=(8, 6))
rc("text", usetex=True)
rc("font", size=16)

with open("snippet/data/N4L1K1shift/analyzer_result.txt") as f:
    reader = csv.reader(f)
    ring4L1K1 = [float(row[0]) for row in reader]
    ring4L1K1_hist, bin_edges = np.histogram(ring4L1K1, range=(0, 15), bins=15 * 4)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

with open("snippet/data/N4L1K10shift/analyzer_result.txt") as f:
    reader = csv.reader(f)
    ring4L1K10 = [float(row[0]) for row in reader]
    ring4L1K10_hist, _ = np.histogram(ring4L1K10, range=(0, 15), bins=15 * 4)


with open("snippet/data/N4L1K50shift/analyzer_result.txt") as f:
    reader = csv.reader(f)
    ring4L1K50 = [float(row[0]) for row in reader]
    ring4L1K50_hist, _ = np.histogram(ring4L1K50, range=(0, 15), bins=15 * 4)

with open("snippet/data/N4L1K100shift/analyzer_result.txt") as f:
    reader = csv.reader(f)
    ring4L1K100 = [float(row[0]) for row in reader]
    ring4L1K100_hist, _ = np.histogram(ring4L1K100, range=(0, 15), bins=15 * 4)

with open("snippet/data/N4L1K10shift/analyzer_result.txt") as f:
    reader = csv.reader(f)
    ringN4L1K10 = [float(row[0]) for row in reader]
    ringN4L1K10_hist, _ = np.histogram(ringN4L1K10, range=(0, 15), bins=15 * 4)

plt.plot(bin_centers, ring4L1K1_hist, label="K\(\pm\)1\% L\(\pm\)1\%")
plt.plot(bin_centers, ring4L1K10_hist, label="K\(\pm\)10\% L\(\pm\)1\%")
plt.plot(bin_centers, ring4L1K50_hist, label="K\(\pm\)50\% L\(\pm\)1\%")
plt.plot(bin_centers, ring4L1K100_hist, label="K\(\pm\)100\% L\(\pm\)1\%")
# plt.plot(bin_centers, ring4L01K0_hist, label="K\(\pm\)0\% L\(\pm\)0.1\%")
plt.ylabel("frequency", size=20)
plt.xlabel("evaluation function value", size=20)
plt.legend(loc="best")
plt.show()
