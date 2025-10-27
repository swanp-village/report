from dataclasses import dataclass
import math
import numpy as np
import numpy.typing as npt
from scipy.optimize import differential_evolution
from cmaes import CMA
from config.random import get_differential_evolution_rng
from MRR.analyzer import analyze
from MRR.evaluator import evaluate_band
from MRR.graph import Graph
from MRR.logger import Logger
from MRR.simulator import (
    calculate_practical_FSR,
    calculate_ring_length,
    calculate_x,
    optimize_N,
)
from MRR.transfer_function import simulate_transfer_function
from scipy.stats.qmc import LatinHypercube
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from cma import CMAEvolutionStrategy
from typing import Tuple
import numpy.typing as npt
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone
from typing import Tuple,List
# ----------------------------------------------------

def optimize_L(
    n_g: float,
    n_eff: float,
    FSR: float,
    center_wavelength: float,
    min_ring_length: float,
    number_of_rings: int,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float_], np.float_]:
    for i in range(100):
        N = optimize_N(
            center_wavelength=center_wavelength,
            min_ring_length=min_ring_length,
            n_eff=n_eff,
            n_g=n_g,
            number_of_rings=number_of_rings,
            FSR=FSR,
            rng=rng,
        )
        L = calculate_ring_length(center_wavelength=center_wavelength, n_eff=n_eff, N=N)
         
        
        practical_FSR = calculate_practical_FSR(center_wavelength=center_wavelength, n_eff=n_eff, n_g=n_g, N=N)
        if practical_FSR > FSR * 0.99 and practical_FSR < FSR * 1.01 and np.all(L < 0.1):
            break
    if i == 99:
        raise Exception("FSR is too strict")

    return N, L, practical_FSR




@dataclass
class OptimizeKParams:
    L: npt.NDArray[np.float_]
    n_g: float
    n_eff: float
    eta: float
    alpha: float
    center_wavelength: float
    length_of_3db_band: float
    FSR: np.float_
    max_crosstalk: float
    H_p: float
    H_s: float
    H_i: float
    r_max: float
    weight: list[float]


def cma_run(initial, bounds_array, popsize, sigma, generations, params,objective_func):
    # bounds_array: shape (N, 2)
    lower_bounds = bounds_array[:, 0]
    upper_bounds = bounds_array[:, 1]

    opts = {
        'bounds': [lower_bounds, upper_bounds],
        'popsize': popsize,
        'verb_log': 0,
        'verbose': -9,
        'tolfun': 0,        # 目的関数値の改善による停止を無効化
        'tolx': 0,          # 探索空間の変化による停止を無効化
        'tolfunhist': 0,    # 過去の履歴による停止を無効化
        'tolflatfitness': 0, # フィットネスが平坦になったことによる停止を無効化
        'maxiter': generations,  
    }

    es = CMAEvolutionStrategy(initial, sigma, opts)

    best_solution = None
    best_fitness = float("inf")

    for generation in range(generations):
        candidates = es.ask()
        fitnesses = [objective_func(x) for x in candidates]
        es.tell(candidates, fitnesses)
     
        if es.sigma < 0.1:
            es.sigma = 0.1
        
        elif es.sigma > 0.7:
            es.sigma = 0.7
       

        min_fit = min(fitnesses)
        if min_fit < best_fitness:
            best_fitness = min_fit
            best_solution = candidates[fitnesses.index(min_fit)]

        # ログ出力（任意）
        if generation % 50 == 0 or generation == generations - 1:
            print(f"Gen {generation}: sigma = {es.sigma:.4f}, best_fitness = {best_fitness:.6f}")


    return best_solution, best_fitness  
    
def normalize_K(K_physical: np.ndarray, eta_max: float) -> np.ndarray:
    """物理スケール [1e-12, eta] から [0, 1] に正規化する"""
    K_min = 1e-12
    K_range = eta_max - K_min
    K_normalized = (K_physical - K_min) / K_range
    return np.clip(K_normalized, 0.0, 1.0)

def denormalize_K(K_normalized: np.ndarray, eta_max: float) -> np.ndarray:
    """正規化スケール [0, 1] から物理スケール [1e-12, eta] に戻す"""
    K_min = 1e-12
    K_range = eta_max - K_min
    K_physical = K_normalized * K_range + K_min
    return np.clip(K_physical, K_min, eta_max)


# --- 【ANNアンサンブル予測関数】 ---
def predict_ensemble(K_2d: np.ndarray, ensemble_models: List[MLPRegressor]) -> Tuple[float, float]:
    """複数のANNモデルで予測を行い、平均(mu)と標準偏差(sigma)を計算する"""
    
    # 各モデルで予測
    predictions = np.array([model.predict(K_2d)[0] for model in ensemble_models])
    
    # 予測の平均を mu に、標準偏差を sigma に設定
    mu = np.mean(predictions)
    sigma = np.std(predictions)
    
    return mu, sigma 


# --- 【獲得関数 (Acquisition Function) - ANNベースに修正】 ---
def acquisition_function(K: npt.NDArray[np.float_], ensemble_models: List[MLPRegressor], best_so_far: float) -> float:
    """ANNアンサンブルの予測値と不確実性を利用した獲得関数"""
    K_2d = K.reshape(1, -1)
    
    # ANNアンサンブルから mu と sigma を取得
    mu, sigma = predict_ensemble(K_2d, ensemble_models)
    
    # 探索と活用のバランスを取るロジック (LCB形式)
    # betaは探索（不確実性）をどれだけ重視するかのパラメータ (FSR 35nmでは高めに設定)
    beta = 15.0 # 探索を強制するため、以前より高い値を推奨
    
    # GPRと同様、μは最小化したい値(-E)を予測
    acquisition_value = mu - beta * sigma 
    
    # CMA-ESに渡すために最小化形式のまま返す
    return acquisition_value


# --- 【既存の cma_run 関数 (SAO内部探索用) の修正】 ---
# CMA-ESが獲得関数を最適化する際に、途中で停止しないように修正
"""
def acquisition_function(K: npt.NDArray[np.float_], gpr_model: GaussianProcessRegressor, best_so_far: float) -> float:
    
    #サロゲートモデルの予測値と不確実性を利用して、次に評価すべき点のスコアを計算する。
    
    #ここでは、単純な「予測値 + 探索ボーナス」で局所解脱出を促す例を示す。
    
    K_2d = K.reshape(1, -1)
    
    # 予測の平均値 (mu) と標準偏差 (sigma/不確実性) を取得
    mu, sigma = gpr_model.predict(K_2d, return_std=True)
    
    # [局所解脱出のためのロジック]
    # 不確実性(sigma)が大きいほどスコアが高くなるようにする（探索を促す）。
    # CMA-ESは minimize を行うため、評価値 E = -F で定義されていると仮定し、
    # 獲得関数は最大化したいので、マイナスをつけて返す。
    
    # 探索と活用のバランスを取るロジックの例 (Upper Confidence Bound的なもの)
    # betaは探索（不確実性）をどれだけ重視するかのパラメータ (FSR 35nmでは高めに設定)
    beta = 10
    
    # GPRの予測値 (mu) を活用しつつ、不確実性 (sigma) を探索ボーナスとして加算
    # 注: optimize_K_funcは -E を返すため、muも最小化したい値のマイナスであると仮定
    
    acquisition_value = mu[0] - beta * sigma[0] 
    
    # CMA-ESに渡すために最小化形式に戻す
    return acquisition_value
"""
def optimize_K(
    eta: float,
    number_of_rings: int,
    rng: np.random.Generator,
    params: OptimizeKParams,
) -> tuple[npt.NDArray[np.float_], float]:
    #-----初期設定-----
    #model
    N_dim = number_of_rings + 1

    num_ann = 8
    hidden_layer_sizes = (100,50,20)
    base_ann_model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=500,
        activation='relu',
        solver='adam',
        random_state=42,
        learning_rate_init=0.01 
    )
    ensemble_models = [clone(base_ann_model) for _ in range(num_ann)]
    initial_samples = N_dim * 10 # 初期サンプル数を10Nに増やす
    MAX_SAO_ITERATIONS = 200 # SAOの総予算
    X_train = [] 
    Y_train = []
    best_K_norm = None
    best_fitness = float("inf")
    #-----データ収集-----
    print("データ収集開始")
    bounds_normalized = np.array([(0.0, 1.0) for _ in range(N_dim)])
    lhs = LatinHypercube(d=number_of_rings + 1, seed=rng)
    #[0, 1]の空間で初期サンプルを生成
    initial_K_samples =  lhs.random(n=initial_samples) 
    for K_sample in initial_K_samples:
        #評価関数で計算
        train_fitness = optimize_K_func(K_sample,params)
        X_train.append(K_sample)
        Y_train.append(train_fitness)

        if train_fitness < best_fitness:
            best_fitness = train_fitness
            best_K = K_sample
    print("finish")
    for iteration in range (MAX_SAO_ITERATIONS):
        X_arr = np.array(X_train)
        Y_arr = np.array(Y_train)
        model.fit(X_arr, Y_arr)
        print(f"STEP 3: SAO Iteration {iteration+1}. モデル訓練完了。")
        
        # 5. 獲得関数 (Acquisition Function) の最適化
        # CMA-ESを呼び出し、獲得関数を最大化（最小化形式のためマイナス）
        
        def acquisition_wrapper(K_candidate):
            # acquisition_function が最大化したい値を返すため、最小化のためにマイナスを付ける
            return -acquisition_function(K_candidate, gpr_model, best_fitness)

        # CMA-ESを使って、獲得関数が最大になるKの候補を探す
        # ここでは既存の cma_run を Acquisition Function の最適化に再利用
        # 注意: generations は短く設定し、高速なサロゲートモデル上での探索に専念させる

        # 2. 初期解をランダムに生成する（フリーズ対策）
        # 以前の「最良点から開始」を止め、ランダムな初期解を生成します。
        initial_random_norm = rng.uniform(1e-12, eta, size=(number_of_rings + 1,))

        acq_best_K, _ = cma_run(
            initial=initial_random_norm, # 最良解の近傍から開始
            bounds_array=bounds_array,
            popsize=10, 
            sigma=0.7, 
            generations=300, 
            params=params,
            objective_func=acquisition_wrapper # 目的関数を獲得関数に置き換え
        )
        
        # 6. 真値の再評価とデータの更新 (モデルの検証)

        
        # 獲得関数が提案した点 (acq_best_K) を元の評価関数で確認
        true_fitness_new = optimize_K_func(acq_best_K, params)
        
        # データセットを更新
        X_train.append(acq_best_K)
        Y_train.append(true_fitness_new)
        
        # 全体の最良解を更新
        if true_fitness_new < best_fitness:
            best_fitness = true_fitness_new
            best_K = acq_best_K
            
        print(f"STEP 4: 真値再評価完了。Best Fitness (True) = {best_fitness:.6f}")
    
    # --- [最終結果] ----------------------------------------------------
    E: float = -best_fitness
    K: npt.NDArray[np.float_] = best_K
    
    return K, E




def optimize(
    n_g: float,
    n_eff: float,
    eta: float,
    alpha: float,
    center_wavelength: float,
    length_of_3db_band: float,
    FSR: float,
    max_crosstalk: float,
    H_p: float,
    H_s: float,
    H_i: float,
    r_max: float,
    weight: list[float],
    min_ring_length: float,
    number_of_rings: int,
    number_of_generations: int,
    strategy: list[float],
    logger: Logger,
    skip_plot: bool = False,
    seedsequence: np.random.SeedSequence = np.random.SeedSequence(),
) -> None:
    rng = get_differential_evolution_rng(seedsequence=seedsequence)
    N_list: list[npt.NDArray[np.int_]] = [np.array([]) for _ in range(number_of_generations)]
    L_list: list[npt.NDArray[np.float_]] = [np.array([]) for _ in range(number_of_generations)]
    K_list: list[npt.NDArray[np.float_]] = [np.array([]) for _ in range(number_of_generations)]
    FSR_list: npt.NDArray[np.float_] = np.zeros(number_of_generations, dtype=np.float_)
    E_list: list[float] = [0 for _ in range(number_of_generations)]
    method_list: list[int] = [0 for _ in range(number_of_generations)]
    best_E_list: list[float] = [0 for _ in range(number_of_generations)]
    analyze_score_list: list[float] = [0 for _ in range(number_of_generations)]
    for m in range(number_of_generations):
        N: npt.NDArray[np.int_]
        L: npt.NDArray[np.float_]
        practical_FSR: np.float_

        kind: npt.NDArray[np.int_]
        counts: npt.NDArray[np.int_]

        if m < 10:
            method: int = 4
        else:
            method = rng.choice([1, 2, 3, 4], p=strategy)

        if method == 1:
            max_index = np.argmax(E_list)
            max_N = N_list[max_index]

            kind, counts = rng.permutation(np.unique(max_N, return_counts=True), axis=1)  # type: ignore
            N = np.repeat(kind, counts)
            L = calculate_ring_length(center_wavelength=center_wavelength, n_eff=n_eff, N=N)
            practical_FSR = calculate_practical_FSR(center_wavelength=center_wavelength, n_eff=n_eff, n_g=n_g, N=N)
        elif method == 2:
            max_index = np.argmax(E_list)
            max_N = N_list[max_index]
            N = rng.permutation(max_N)
            L = calculate_ring_length(center_wavelength=center_wavelength, n_eff=n_eff, N=N)
            practical_FSR = calculate_practical_FSR(center_wavelength=center_wavelength, n_eff=n_eff, n_g=n_g, N=N)
        elif method == 3:
            max_index = np.argmax(E_list)
            max_N = N_list[max_index]
            kind = np.unique(max_N)  # type: ignore
            N = rng.choice(kind, number_of_rings)
            while not set(kind) == set(N):
                N = rng.choice(kind, number_of_rings)
            L = calculate_ring_length(center_wavelength=center_wavelength, n_eff=n_eff, N=N)
            practical_FSR = calculate_practical_FSR(center_wavelength=center_wavelength, n_eff=n_eff, n_g=n_g, N=N)
        else:
            N, L, practical_FSR = optimize_L(
                n_g=n_g,
                n_eff=n_eff,
                FSR=FSR,
                center_wavelength=center_wavelength,
                min_ring_length=min_ring_length,
                number_of_rings=number_of_rings,
                rng=rng,
            )
        #N = [78,78,78,117,117,117] #6th
        N = [88,88,110,110,110,110]
        #N = [110,110,88,88,88,88,110,110]
        #N = [78,78,78,468,468,117,117,117] #8th
        #N = [117,117,117,156,156,156,117,117,156,156] #10th
        #N = [117,117,117,117,468,468,468,78,78,78,78,78] #12th
        #N = [117,117,117,117,468,468,468,468,78,78,78,78,78,78] #14th
        
        L = calculate_ring_length(center_wavelength=center_wavelength, n_eff=n_eff, N=N)
        normal_evaluations = []
        perturbed_evaluations = []
       
        K, E = optimize_K(
            eta=eta,
            number_of_rings=number_of_rings,
            rng=rng,
            params=OptimizeKParams(
                L=L,
                n_g=n_g,
                n_eff=n_eff,
                eta=eta,
                alpha=alpha,
                center_wavelength=center_wavelength,
                length_of_3db_band=length_of_3db_band,
                FSR=practical_FSR,
                max_crosstalk=max_crosstalk,
                H_p=H_p,
                H_s=H_s,
                H_i=H_i,
                r_max=r_max,
                weight=weight,
            ),
        )
        N_list[m] = N
        L_list[m] = L
        FSR_list[m] = practical_FSR
        K_list[m] = K
        E_list[m] = E
        analyze_score = 0.0
        if E > 20:
            for L_error_rate, K_error_rate in zip([0.01, 0.1, 1, 10], [1, 10, 100]):
                analyze_result = analyze(
                    n=1000,
                    L_error_rate=L_error_rate,
                    K_error_rate=K_error_rate,
                    L=L,
                    K=K,
                    n_g=n_g,
                    n_eff=n_eff,
                    eta=eta,
                    alpha=alpha,
                    center_wavelength=center_wavelength,
                    length_of_3db_band=length_of_3db_band,
                    FSR=FSR,
                    max_crosstalk=max_crosstalk,
                    H_p=H_p,
                    H_s=H_s,
                    H_i=H_i,
                    r_max=r_max,
                    weight=weight,
                    min_ring_length=min_ring_length,
                    seedsequence=seedsequence,
                    skip_plot=True,
                    logger=logger,
                )
                if analyze_result > 0.5:
                    analyze_score += 1
            analyze_score_list[m] = analyze_score
        best_index = np.argmax(E_list)
        best_N = N_list[best_index]
        best_L = L_list[best_index]
        best_K = K_list[best_index]
        best_FSR = FSR_list[best_index]
        best_E = E_list[best_index]
        best_analyze_score = analyze_score_list[best_index]
        print(m + 1)
        logger.print_parameters(K=K, L=L, N=N, FSR=practical_FSR, E=E, analyze_score=analyze_score, format=True)
        print("==best==")
        logger.print_parameters(
            K=best_K, L=best_L, N=best_N, FSR=best_FSR, E=best_E, analyze_score=best_analyze_score, format=True
        )
        
        print("================")

        method_list[m] = method
        best_E_list[m] = best_E

    max_index = np.argmax(E_list)
    result_N = N_list[max_index]
    result_L = L_list[max_index]
    result_K = K_list[max_index]
    result_FSR = FSR_list[max_index]
    result_E = E_list[max_index]
    result_analyze_score = analyze_score_list[max_index]
    x = calculate_x(center_wavelength=center_wavelength, FSR=result_FSR)
    y = simulate_transfer_function(
        wavelength=x,
        L=result_L,
        K=result_K,
        alpha=alpha,
        eta=eta,
        n_eff=n_eff,
        n_g=n_g,
        center_wavelength=center_wavelength,
    )
    print("result")
    
    logger.print_parameters(
        K=result_K, L=result_L, N=result_N, FSR=result_FSR, E=result_E, analyze_score=result_analyze_score
    )

    logger.save_result(L=result_L, K=result_K)
    print("save data")
    logger.save_DE_data(
        N_list=N_list,
        L_list=L_list,
        K_list=K_list,
        FSR_list=FSR_list,
        E_list=E_list,
        method_list=method_list,
        best_E_list=best_E_list,
        analyze_score_list=analyze_score_list,
    )
    print("end")
    if E > 0 and not skip_plot:
        graph = Graph()
        graph.create()
        graph.plot(x, y)
        graph.show(logger.generate_image_path())
        plt.figure()
        plt.plot(x, y)
        plt.title("Transfer Function")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Transmittance (dB)")
        plt.grid(True)
        plt.show()


def optimize_K_func(K: npt.NDArray[np.float_], params: OptimizeKParams) -> np.float_:
    
    

    x = calculate_x(center_wavelength=params.center_wavelength, FSR=params.FSR)
    y = simulate_transfer_function(
        wavelength=x,
        L=params.L,
        K=K,
        alpha=params.alpha,
        eta=params.eta,
        n_eff=params.n_eff,
        n_g=params.n_g,
        center_wavelength=params.center_wavelength,
    )
    #print(f"x: {x}")
    #print(f"y: {y}")
    

    

    return -evaluate_band(
        x=x,
        y=y,
        center_wavelength=params.center_wavelength,
        length_of_3db_band=params.length_of_3db_band,
        max_crosstalk=params.max_crosstalk,
        H_p=params.H_p,
        H_s=params.H_s,
        H_i=params.H_i,
        r_max=params.r_max,
        weight=params.weight,
        ignore_binary_evaluation=False,
    )
    #print(f"Fitness value: {fitness}")
"""
def optimize_perturbed_K_func(K: npt.NDArray[np.float_], params: OptimizeKParams) -> tuple[float, float]:

    
    #誤差として結合率 K に +0.005 および -0.005 を適用した場合の評価値を計算。

    #Parameters:
    #- K: 結合率の配列
    #- params: 最適化パラメータ

    #Returns:
   # - E_positive: +0.005 の誤差を加えた場合の評価値
    #- E_negative: -0.005 の誤差を加えた場合の評価値
    

    # 正の誤差を加える
    perturbed_K_positive = np.clip(K + 0.005, 1e-12, params.eta)

    # 負の誤差を加える
    perturbed_K_negative = np.clip(K - 0.005, 1e-12, params.eta)

    # 波長を計算
    x = calculate_x(center_wavelength=params.center_wavelength, FSR=params.FSR)

    # 正の誤差での評価値
    y_positive = simulate_transfer_function(
        wavelength=x,
        L=params.L,
        K=perturbed_K_positive,
        alpha=params.alpha,
        eta=params.eta,
        n_eff=params.n_eff,
        n_g=params.n_g,
        center_wavelength=params.center_wavelength,
    )
    E_positive = -evaluate_band(
        x=x,
        y=y_positive,
        center_wavelength=params.center_wavelength,
        length_of_3db_band=params.length_of_3db_band,
        max_crosstalk=params.max_crosstalk,
        H_p=params.H_p,
        H_s=params.H_s,
        H_i=params.H_i,
        r_max=params.r_max,
        weight=params.weight,
        ignore_binary_evaluation=False,
    )

    # 負の誤差での評価値
    y_negative = simulate_transfer_function(
        wavelength=x,
        L=params.L,
        K=perturbed_K_negative,
        alpha=params.alpha,
        eta=params.eta,
        n_eff=params.n_eff,
        n_g=params.n_g,
        center_wavelength=params.center_wavelength,
    )
    E_negative = -evaluate_band(
        x=x,
        y=y_negative,
        center_wavelength=params.center_wavelength,
        length_of_3db_band=params.length_of_3db_band,
        max_crosstalk=params.max_crosstalk,
        H_p=params.H_p,
        H_s=params.H_s,
        H_i=params.H_i,
        r_max=params.r_max,
        weight=params.weight,
        ignore_binary_evaluation=False,
    )
 

    return E_positive, E_negative
    """

