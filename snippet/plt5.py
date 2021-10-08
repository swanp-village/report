import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

fig, ax = plt.subplots(figsize=(8, 6))
rc("text", usetex=True)
rc("font", size=16)

with open("snippet/data/analyzer_result_10th.txt") as f:
    reader = csv.reader(f)
    ring10 = [float(row[0]) for row in reader]
    ring10_hist, bin_edges = np.histogram(ring10, range=(0, 15), bins=15 * 4)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

with open("snippet/data/analyzer_result_8th.txt") as f:
    reader = csv.reader(f)
    ring8 = [float(row[0]) for row in reader]
    ring8_hist, _ = np.histogram(ring8, range=(0, 15), bins=15 * 4)


with open("snippet/data/analyzer_result_6th.txt") as f:
    reader = csv.reader(f)
    ring6 = [float(row[0]) for row in reader]
    ring6_hist, _ = np.histogram(ring6, range=(0, 15), bins=15 * 4)

with open("snippet/data/analyzer_result_4th.txt") as f:
    reader = csv.reader(f)
    ring4 = [float(row[0]) for row in reader]
    ring4_hist, _ = np.histogram(ring4, range=(0, 15), bins=15 * 4)

# plt.hist(ring10, range=(0, 15), bins=15 * 4, label="10th", histtype="step")
# plt.hist(ring8, range=(0, 15), bins=15 * 4, label="8th", histtype="step")
# plt.hist(ring6, range=(0, 15), bins=15 * 4, label="6th", histtype="step")
# plt.hist(ring4, range=(0, 15), bins=15 * 4, label="4th", histtype="step")
plt.plot(bin_centers, ring4_hist, label="4th")
plt.plot(bin_centers, ring6_hist, label="6th")
plt.plot(bin_centers, ring8_hist, label="8th")
plt.plot(bin_centers, ring10_hist, label="10th")
plt.ylabel("frequency", size=20)
plt.xlabel("evaluation function value", size=20)
plt.legend(loc="upper right")
plt.show()
