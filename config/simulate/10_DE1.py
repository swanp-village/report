import numpy as np
import matplotlib.pyplot as plt

# 設定パラメータ
dose = 1.5e15  # ドーズ量 (cm^-2)

# PのSRIMデータ
rp_p_data = {
    20: 30.0,
    50: 69.0,
    100: 134.2
}
delta_rp_p_data = {
    20: 13.6,
    50: 27.7,
    100: 40.7
}

# AsのSRIMデータ
rp_as_data = {
    20: 19.6,
    50: 38.8,
    100: 69.2
}
delta_rp_as_data = {
    20: 6.9,
    50: 12.4,
    100: 20.6
}

#プロット範囲
x_nm = np.linspace(0, 250, 500) # 0nmから250nmまで、500点
x_cm = x_nm * 1e-7 # xをcmに変換

# エネルギーのリスト
energies_to_plot = [20, 50, 100]

# --- リン (P) のグラフ ---
plt.figure(figsize=(10, 6))
for energy in energies_to_plot:
    # PのRpとDelta_Rpを取得
    Rp_P = rp_p_data[energy]
    Delta_Rp_P = delta_rp_p_data[energy]

    # 計算のためにnmをcmに変換
    Rp_P_cm = Rp_P * 1e-7
    Delta_Rp_P_cm = Delta_Rp_P * 1e-7

    # 濃度を計算
    C_max_P = dose / (np.sqrt(2 * np.pi) * Delta_Rp_P_cm)
    concentration_P = C_max_P * np.exp(-((x_cm - Rp_P_cm)**2) / (2 * Delta_Rp_P_cm**2))

    # プロット
    plt.plot(x_nm, concentration_P, label=f' {energy} keV')

plt.xlabel('Depth (nm)')
plt.ylabel('Concentration (atoms/cm$^3$)')
plt.title('P')
plt.yscale('log') # y軸を対数スケールに設定
plt.grid(True, which="both", ls="--", c='0.7')
plt.legend()
plt.xlim(0, 250) # x軸の範囲
plt.ylim(1e14, 1e21) # y軸の範囲
