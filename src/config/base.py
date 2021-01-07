config = {
    'eta': 1,  # 結合損
    'alpha': 52.96,  # 伝搬損失係数
    'n_eff': 3.3938,  # 等価屈折率
    'n_g': 4.2,  # 実行屈折率
    'number_of_rings': 8,  # リング次数
    'center_wavelength': 1550e-9,  # 中心波長
    'FSR': 20e-9,
    'min_ring_length': 50e-6,

    'max_loss_in_pass_band': -30,
    'required_loss_in_stop_band': -60,
    'length_of_3db_band': 1e-9,

    'number_of_episodes_in_L': 20,
    'number_of_episodes_in_K': 3,
    'number_of_steps': 1000,
}
