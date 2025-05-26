import numpy as np
import numpy.typing as npt
from scipy.signal import argrelmax, argrelmin


def evaluate_band(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    center_wavelength: float,
    length_of_3db_band: float,
    max_crosstalk: float,
    H_p: float,
    H_s: float,
    H_i: float,
    r_max: float,
    weight: list[float],
    ignore_binary_evaluation: bool = False,
) -> np.float_:
    pass_band, cross_talk = _get_pass_band(x=x, y=y, H_p=H_p, center_wavelength=center_wavelength)
    if pass_band.shape[0] == 1:
        start = pass_band[0][0]
        end = pass_band[0][1]

    else:
        return np.float_(0)

    result = [
        _evaluate_pass_band(x=x, y=y, H_p=H_p, start=start, end=end),
        _evaluate_stop_band(x=x, y=y, H_p=H_p, H_s=H_s, start=start, end=end),
        _evaluate_insertion_loss(x=x, y=y, H_i=H_i, center_wavelength=center_wavelength),
        _evaluate_3db_band(x=x, y=y, length_of_3db_band=length_of_3db_band, start=start, end=end),
        _evaluate_ripple(x=x, y=y, r_max=r_max, start=start, end=end),
        _evaluate_cross_talk(y=y, max_crosstalk=max_crosstalk, pass_band_start=start, pass_band_end=end),
        _evaluate_shape_factor(x=x, y=y, start=start, end=end),
    ]
    n_eval = len(result)
    W_c = weight[:n_eval]
    W_b = weight[n_eval:]
    E_c = np.float_(0)
    E_b = np.float_(1)
    for i in range(n_eval):
        E_c += result[i][0] * W_c[i]
        if not result[i][1]:
            E_b *= W_b[i]
    if ignore_binary_evaluation:
        return E_c
    E = E_c * E_b
    # 各評価関数の結果を表示
    """
    print(E_b)
    print(E_c)
    for i, res in enumerate(result):
        print(f"評価関数 {i+1}: 値 = {res[0]}, バイナリ評価 = {res[1]}")
    """

    return E
    


def _calculate_pass_band_range(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    H_p: float,
) -> npt.NDArray[np.int_]:
    start = 0
    end = x.size - 1
    a: npt.NDArray[np.bool_] = np.where(y <= H_p, True, False)
    b: npt.NDArray[np.bool_] = np.append(a[1:], a[-1])
    pass_band_range: npt.NDArray[np.int_] = np.where(np.logical_xor(a, b))[0]
    if pass_band_range.size == 0:
        return pass_band_range
    if not a[pass_band_range][0]:
        pass_band_range = np.append(start, pass_band_range)
    if a[pass_band_range][-1]:
        pass_band_range = np.append(pass_band_range, end)
    pass_band_range = np.reshape(pass_band_range, [pass_band_range.size // 2, 2])

    return pass_band_range


def _get_pass_band(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    H_p: float,
    center_wavelength: float,
) -> (npt.ArrayLike, npt.ArrayLike):
    pass_band = []
    cross_talk = []
    for start, end in _calculate_pass_band_range(x=x, y=y, H_p=H_p):
        if center_wavelength >= x[start] and center_wavelength <= x[end]:
            pass_band.extend([start, end])
        else:
            cross_talk.extend([start, end])

    pass_band = np.reshape(pass_band, [len(pass_band) // 2, 2])
    cross_talk = np.reshape(cross_talk, [len(cross_talk) // 2, 2])

    return pass_band, cross_talk


def _get_3db_band1(x: npt.NDArray[np.float_], y: npt.NDArray[np.float_], start: int, end: int) -> npt.ArrayLike:
    border: np.float_ = y.max() - 3
    a = np.where(y[start:end] <= border, True, False)
    b = np.append(a[1:], a[-1])
    index = np.where(np.logical_xor(a, b))[0]

    return index


def _get_3db_band(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    start: int,
    end: int
) -> npt.NDArray[np.int_]:
    # dBスケール前提（最大値から3dB以内）
    y_db = y[start:end]
    threshold = y_db.max() - 3

    # 条件を満たすインデックス
    mask = y_db >= threshold
    idx = np.where(mask)[0]

    if idx.size == 0:
        return np.array([], dtype=int)

    # 通過帯域の範囲（インデックスそのものを返す）
    return idx

def _evaluate_pass_band(
    x: npt.NDArray[np.float_], y: npt.NDArray[np.float_], H_p: float, start: int, end: int
) -> tuple[np.float_, bool]:
    distance: np.float_ = x[1] - x[0]
    a = abs(H_p * (x[end] - x[start]))
    if a < 1e-9:
        penalty = np.exp(-a/1e-9)
        return (np.float_(penalty),False)
    b = abs(np.sum(H_p - y[start:end]) * distance)
    E = b / a
    return (E, True)


def _evaluate_stop_band(
    x: npt.NDArray[np.float_], y: npt.NDArray[np.float_], H_p: float, H_s: float, start: int, end: int
) -> tuple[np.float_, bool]:
    distance: np.float_ = x[1] - x[0]
    c = abs((H_s - H_p) * ((x[start] - x[0]) + (x[-1] - x[end])))

    if c < 1e-9:
        penalty = np.exp(-c/1e-9)
        return (np.float_(penalty),False)

    y1 = np.where(y[0:start] > H_s, H_p - y[0:start], H_p - H_s)
    y1 = np.where(y1 > 0, y1, 0)
    y2 = np.where(y[end:-1] > H_s, H_p - y[end:-1], H_p - H_s)
    y2 = np.where(y2 > 0, y2, 0)
    d = abs((np.sum(y1) + np.sum(y2)) * distance)
    E = d / c

    return (E, True)


def _evaluate_insertion_loss(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    H_i: float,
    center_wavelength: float,
) -> tuple[np.float_, bool]:
    insertion_loss = y[x == center_wavelength]
    if insertion_loss.size == 0:
        return (np.float_(1e-6), False)

    if insertion_loss[0] < H_i:
        E = H_i / (insertion_loss[0] + 1e-6)  # ペナルティ計算を連続化
    else:
        E = 1 - insertion_loss[0] / H_i

    return (E, True)


def _evaluate_3db_band(
    x: npt.NDArray[np.float_], y: npt.NDArray[np.float_], length_of_3db_band: float, start: int, end: int
) -> tuple[np.float_, bool]:
    distance: np.float_ = x[1] - x[0]
    index = _get_3db_band(x=x, y=y, start=start, end=end)
    if index.size <= 1:
        # ペナルティ計算を追加
        return (np.float_(0), False)

    practical_length_of_3db_band = distance * (index[-1] - index[0])
    if practical_length_of_3db_band > length_of_3db_band:
        E = (2 * length_of_3db_band - practical_length_of_3db_band) / length_of_3db_band
    else:
        E = practical_length_of_3db_band / length_of_3db_band
    E = E ** 3
    return (E, True)
"""
#最大最小型
def _evaluate_ripple(
    x: npt.NDArray[np.float_], y: npt.NDArray[np.float_], r_max: float, start: int, end: int
) -> tuple[np.float_, bool]:
    pass_band = y[start:end]
    index = _get_3db_band(x=x, y=y, start=start, end=end)
    if index.size <= 1:
        return (np.float_(0), False)
    three_db_band = pass_band[index[0] : index[-1]]
    maxid = argrelmax(three_db_band, order=1)
    minid = argrelmin(three_db_band, order=1)
    peak_max = three_db_band[maxid]
    peak_min = three_db_band[minid]
    if len(peak_min) == 0:
        return (1, True)
    dif = peak_max.max() - peak_min.min()
    if dif > r_max:
        E = 0
    else:
        E = 1 - dif / r_max

    return (E, True)

#条件分岐型
def _evaluate_ripple(
    x: npt.NDArray[np.float_], y: npt.NDArray[np.float_], r_max: float, start: int, end: int
) -> tuple[np.float_, bool]:
    pass_band = y[start:end]
    index = _get_3db_band(x=x, y=y, start=start, end=end)
    if index.size <= 1:
        # ペナルティ計算
        return (np.float_(0), False)

    three_db_band = pass_band[index[0] : index[-1]]
   
    maxid = argrelmax(three_db_band, order=1)
    minid = argrelmin(three_db_band, order=1)
   
    if len(minid[0])==0 or len(maxid[0])==0:
        peak_max = three_db_band.max()
        peak_min = three_db_band.min()
    else:
        peak_max = three_db_band[maxid].max()
        peak_min = three_db_band[minid].min()

    dif = peak_max.max() - peak_min.min()
    if dif > r_max:
        E = 0
    else:
        E = 1 - dif / r_max

    return (E, True)
"""
#標準偏差型
def _evaluate_ripple(
    x: npt.NDArray[np.float_], y: npt.NDArray[np.float_], r_max: float, start: int, end: int
) -> tuple[np.float_, bool]:
    pass_band = y[start:end]
    index = _get_3db_band(x=x, y=y, start=start, end=end)
    if index.size <= 3:
        print("すくない",index)
        return (np.float_(0), False)
        # ペナルティ計算

    n = index.size
    if n > 10:  # 十分な点数あるとき
        central_index = index[int(0.1 * n) : int(0.9 * n)]
        three_db_band = pass_band[central_index]
    else:
        three_db_band = pass_band[index]  # 点数少ない時はそのまま使う
        
    std_ripple = np.std(three_db_band)
    range_ripple = three_db_band.max() - three_db_band.min()

 
    r_max1 = 1.0

    if std_ripple > r_max1 or range_ripple > r_max:
        E = 0
    else:
        E = 1 - (std_ripple + range_ripple) / (r_max1 * r_max)

    return (np.float_(E), True)



def _evaluate_cross_talk(
    y: npt.NDArray[np.float_], max_crosstalk: float, pass_band_start: int, pass_band_end: int
) -> tuple[np.float_, bool]:
    start = y[:pass_band_start]
    end = y[pass_band_end:]
    maxid_start = np.append(0, argrelmax(start))
    maxid_end = np.append(argrelmax(end), -1)
    start_peak = start[maxid_start]
    end_peak = end[maxid_end]
    excess_start = np.maximum(start_peak - max_crosstalk, 0)
    excess_end = np.maximum(end_peak - max_crosstalk, 0)
    penalty = np.sum(excess_start) + np.sum(excess_end)
    return (1 / (1 + penalty), True)


def _evaluate_shape_factor(
    x: npt.NDArray[np.float_], y: npt.NDArray[np.float_], start: int, end: int
) -> tuple[np.float_, bool]:
    index = _get_3db_band1(x=x, y=y, start=start, end=end)
    if index.size <= 1:
        return (np.float_(1e-6), False)

    practical_length = x[index[-1]] - x[index[0]]
    total_length = x[end] - x[start]
    E = practical_length / total_length
    if E < 0.5:
        return (E ** 2, False)  # ペナルティを連続化
    return (E, True)

