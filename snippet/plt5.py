import csv

import matplotlib.pyplot as plt
from matplotlib import rc

fig, ax = plt.subplots(figsize=(8, 6))
rc("text", usetex=True)
rc("font", size=16)

with open("snippet/data/analyzer_result_10th.txt") as f:
    reader = csv.reader(f)
    ring10 = [float(row[0]) for row in reader]

with open("snippet/data/analyzer_result_8th.txt") as f:
    reader = csv.reader(f)
    ring8 = [float(row[0]) for row in reader]

with open("snippet/data/analyzer_result_6th.txt") as f:
    reader = csv.reader(f)
    ring6 = [float(row[0]) for row in reader]

with open("snippet/data/analyzer_result_4th.txt") as f:
    reader = csv.reader(f)
    ring4 = [float(row[0]) for row in reader]

plt.hist(ring10, range=(0, 15), bins=15 * 4, alpha=0.5, label="10th")
plt.hist(ring8, range=(0, 15), bins=15 * 4, alpha=0.5, label="8th")
plt.hist(ring6, range=(0, 15), bins=15 * 4, alpha=0.5, label="6th")
plt.hist(ring4, range=(0, 15), bins=15 * 4, alpha=0.5, label="4th")
plt.ylabel("frequency", size=20)
plt.xlabel("evaluation function value", size=20)
plt.legend(loc="upper right")
plt.show()
