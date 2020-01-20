config = {
    'eta': 1,  # 結合損
    'alpha': 52.96,  # 伝搬損失係数
    'n': 3.3938,  # 屈折率
    'number_of_rings': 2,  # リング次数
    'center_wavelength': 1550e-9,  # 中心波長
    'FSR': 200e-9,

    'max_loss_in_pass_band': -5,
    'required_loss_in_stop_band': -15,

    'number_of_episodes': 10,
    'number_of_steps': 10,
    'learning_rate': 0.1,
    'discount_rate': 0.9,
}
