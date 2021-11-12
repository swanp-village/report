import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

fig, ax = plt.subplots(figsize=(8, 6))
rc("text", usetex=True)
rc("font", size=16)

with open("snippet/data/N4L1K0/analyzer_result.txt") as f:
    reader = csv.reader(f)
    ring4L1K0 = [float(row[0]) for row in reader]
    ring4L1K0_hist, bin_edges = np.histogram(ring4L1K0, range=(0, 15), bins=15 * 4)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

with open("snippet/data/N4L01K0/analyzer_result.txt") as f:
    reader = csv.reader(f)
    ring4L01K0 = [float(row[0]) for row in reader]
    ring4L01K0_hist, _ = np.histogram(ring4L01K0, range=(0, 15), bins=15 * 4)


with open("snippet/data/N4L001K0/analyzer_result.txt") as f:
    reader = csv.reader(f)
    ring4L001K0 = [float(row[0]) for row in reader]
    ring4L001K0_hist, _ = np.histogram(ring4L001K0, range=(0, 15), bins=15 * 4)

# with open("snippet/data/N4L0001K0/analyzer_result.txt") as f:
#     reader = csv.reader(f)
#     ring4L0001K0 = [float(row[0]) for row in reader]
#     ring4L0001K0_hist, _ = np.histogram(ring4L0001K0, range=(0, 15), bins=15 * 4)

# with open("snippet/data/N4L00001K0/analyzer_result.txt") as f:
#     reader = csv.reader(f)
#     ring4L00001K0 = [float(row[0]) for row in reader]
#     ring4L00001K0_hist, _ = np.histogram(ring4L00001K0, range=(0, 15), bins=15 * 4)

# with open("snippet/data/N4L000001K0/analyzer_result.txt") as f:
#     reader = csv.reader(f)
#     ring4L000001K0 = [float(row[0]) for row in reader]
#     ring4L000001K0_hist, _ = np.histogram(ring4L000001K0, range=(0, 15), bins=15 * 4)

plt.plot(bin_centers, ring4L1K0_hist, label="4th L:1%")
plt.plot(bin_centers, ring4L01K0_hist, label="4th L:0.1%")
plt.plot(bin_centers, ring4L001K0_hist, label="4th L:0.01%")
# plt.plot(bin_centers, ring4L0001K0_hist, label="4th L:0.001%")
# plt.plot(bin_centers, ring4L00001K0_hist, label="4th L:0.0001%")
# plt.plot(bin_centers, ring4L000001K0_hist, label="4th L:0.00001%")
plt.ylabel("frequency", size=20)
plt.xlabel("evaluation function value", size=20)
plt.legend(loc="best")
plt.show()
