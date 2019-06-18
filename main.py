import matplotlib.pyplot as plt
import numpy as np
from MRR import MRR


# 結合損
eta = 1

# 減衰定数
a = np.array([1, 1])

# 結合効率
K = np.array([0.65, 0.17, 0.43])

# リング周長
L = np.array([2.284e-4, 2.284e-4])

# 屈折率
n = 3.3938

# 波長レンジ
l = np.arange(1545e-9, 1555e-9, 1e-12)

mrr = MRR(
    eta,
    n,
    a,
    K,
    L
)
y = 20 * np.log10(np.abs(mrr.simulate(l)))

plt.semilogx(l*1e9, y)
plt.xlabel('Wavelength[nm]')
plt.ylabel('Drop Port Power [dB]')
plt.title('{} order MRR (same perimeter)'.format(L.size))
plt.axis([None, None, None, 5])
plt.grid()
plt.show()
