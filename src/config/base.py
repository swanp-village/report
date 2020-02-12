config = {
    'eta': 1,  # 結合損
    'alpha': 52.96,  # 伝搬損失係数
    'n': 3.3938,  # 屈折率
    'number_of_rings': 2,  # リング次数
    'center_wavelength': 1550e-9,  # 中心波長
    'FSR': 50e-9,

    'min_ring_length': 20e-6,
    'max_loss_in_pass_band': -10,
    'required_loss_in_stop_band': -20,
    '3db_band': 1e-9,

    'number_of_episodes_in_L': 50,
    'number_of_episodes_in_K': 50,
    'number_of_steps': 100,
}
