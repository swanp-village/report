import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

fig, ax = plt.subplots(figsize=(8, 6))
rc("text", usetex=True)
rc("font", size=16)

with open("snippet/data/N10L1K10/analyzer_result.txt") as f:
    reader = csv.reader(f)
    ring10L1K10 = [float(row[0]) for row in reader]
    ring10L1K10_hist, bin_edges = np.histogram(ring10L1K10, range=(0, 15), bins=15 * 4)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

with open("snippet/data/N8L1K10/analyzer_result.txt") as f:
    reader = csv.reader(f)
    ring8L1K10 = [float(row[0]) for row in reader]
    ring8L1K10_hist, _ = np.histogram(ring8L1K10, range=(0, 15), bins=15 * 4)


with open("snippet/data/N6L1K10/analyzer_result.txt") as f:
    reader = csv.reader(f)
    ring6L1K10 = [float(row[0]) for row in reader]
    ring6L1K10_hist, _ = np.histogram(ring6L1K10, range=(0, 15), bins=15 * 4)

with open("snippet/data/N4L1K10/analyzer_result.txt") as f:
    reader = csv.reader(f)
    ring4L1K10 = [float(row[0]) for row in reader]
    ring4L1K10_hist, _ = np.histogram(ring4L1K10, range=(0, 15), bins=15 * 4)

with open("snippet/data/N10L10K10/analyzer_result.txt") as f:
    reader = csv.reader(f)
    ring10L10K10 = [float(row[0]) for row in reader]
    ring10L10K10_hist, bin_edges = np.histogram(ring10L10K10, range=(0, 15), bins=15 * 4)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

with open("snippet/data/N8L10K10/analyzer_result.txt") as f:
    reader = csv.reader(f)
    ring8L10K10 = [float(row[0]) for row in reader]
    ring8L10K10_hist, _ = np.histogram(ring8L10K10, range=(0, 15), bins=15 * 4)


with open("snippet/data/N6L10K10/analyzer_result.txt") as f:
    reader = csv.reader(f)
    ring6L10K10 = [float(row[0]) for row in reader]
    ring6L10K10_hist, _ = np.histogram(ring6L10K10, range=(0, 15), bins=15 * 4)

with open("snippet/data/N4L10K10/analyzer_result.txt") as f:
    reader = csv.reader(f)
    ring4L10K10 = [float(row[0]) for row in reader]
    ring4L10K10_hist, _ = np.histogram(ring4L10K10, range=(0, 15), bins=15 * 4)

# plt.hist(ring10, range=(0, 15), bins=15 * 4, label="10th", histtype="step")
# plt.hist(ring8, range=(0, 15), bins=15 * 4, label="8th", histtype="step")
# plt.hist(ring6, range=(0, 15), bins=15 * 4, label="6th", histtype="step")
# plt.hist(ring4, range=(0, 15), bins=15 * 4, label="4th", histtype="step")
plt.plot(bin_centers, ring4L1K10_hist, label="4th L:1%")
# plt.plot(bin_centers, ring6L1K10_hist, label="6th")
# plt.plot(bin_centers, ring8L1K10_hist, label="8th")
plt.plot(bin_centers, ring10L1K10_hist, label="10th L:1%")
plt.plot(bin_centers, ring4L10K10_hist, label="4th L:10%")
# plt.plot(bin_centers, ring6L10K10_hist, label="6th")
# plt.plot(bin_centers, ring8L10K10_hist, label="8th")
plt.plot(bin_centers, ring10L10K10_hist, label="10th L:10%")
plt.ylabel("frequency", size=20)
plt.xlabel("evaluation function value", size=20)
plt.legend(loc="upper right")
plt.show()
