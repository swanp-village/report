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
from scipy import stats
from collections import deque
import random


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
"""
normal_evaluations = []
perturbed_evaluations = []
def combined_evaluation(K: npt.NDArray[np.float_], params: OptimizeKParams) -> float:
    
    #誤差の正負両方を考慮した総合評価値を計算。

   # Parameters:
    #- K: 結合率の配列
    #- params: 最適化パラメータ

    #Returns:
    #- total_score: 総合評価値
    
    global normal_evaluations, perturbed_evaluations
    

    # 通常の評価値
    E_optimal = optimize_K_func(K, params)

    # 正負の誤差による評価値
    E_positive, E_negative = optimize_perturbed_K_func(K, params)

    # 評価値を記録
    normal_evaluations.append(E_optimal)
    perturbed_evaluations.append((E_positive, E_negative))

    # 評価値の統合
    delta_E_positive = abs(E_optimal - E_positive)
    delta_E_negative = abs(E_optimal - E_negative)

    # 総合評価値 (例: 平均変動量をペナルティとして加算)
    total_score = E_optimal + (delta_E_positive + delta_E_negative) / 2

    return total_score

def cma_run(initial, bounds_array, popsize, sigma, generations, params):
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
        fitnesses = [optimize_K_func(x,params) for x in candidates]
        es.tell(candidates, fitnesses)
        min_fit = min(fitnesses)
        if min_fit < best_fitness:
            print("best_fitness",best_fitness)
            #print("min_fitness",min_fit)
            best_fitness = min_fit
            #print("new_best",best_fitness)
            best_solution = candidates[fitnesses.index(min_fit)]

        # ログ出力（任意）
        #if generation % 50 == 0 or generation == generations - 1:
            #print(f"Gen {generation}: sigma = {es.sigma:.4f}, best_fitness = {best_fitness:.6f}")


    return best_solution, best_fitness  
"""
#CMA-ES動作コード_pycma
def SHACMA_run(initial, bounds_array, popsize, sigma, generations, params):
    # bounds_array: shape (N, 2)
    lower_bounds = bounds_array[:, 0]
    upper_bounds = bounds_array[:, 1]
    xdim = len(initial)
    init_ccov1 = 1.0 / (xdim**1.5)
    init_ccovmu = 1.0 / (xdim**1.5)


    H = 100 # メモリサイズ
    mem_sigma = deque([sigma] * H, maxlen = H) # 探索範囲のメモリ
    mem_ccov = deque([(init_ccov1, init_ccovmu)] * H, maxlen = H) # 学習率のメモリ
    archive = deque(maxlen = popsize * 2) # 外部アーカイブ
    k = 0 # メモリ更新回数

    opts = {
        'bounds': [lower_bounds, upper_bounds],
        'popsize': popsize,
        'verb_log': 0,
        'verbose': -9,  # suppress internal logs
        'tolfun':1e-13,
    }

    es = CMAEvolutionStrategy(initial, sigma, opts)

    best_solution = None
    #best_fitness = float("inf")
    best_fitness = 1.0
    stagnation_counter = 0
    counter = 0

    for generation in range (generations):
        #---成功履歴からパラメータを摘出---
        m_idx = np.random.randint(0, H)
        base_c1,base_cmu = mem_ccov[m_idx]
        curr_sigma = stats.cauchy.rvs(loc = mem_sigma[m_idx], scale = 0.1)# sigmaに関して少しの揺れを加える
        curr_ccov1 = stats.cauchy.rvs(loc = base_c1, scale = 0.01)# ccovに関して少しの揺れを加える
        curr_ccovmu = stats.cauchy.rvs(loc = base_cmu, scale = 0.01)
     
        if counter < 10:
            curr_sigma = 0.996
            curr_ccov1 =  1.0 / (xdim**1.5)
            curr_ccovmu = 0.01 * curr_ccov1
            counter  += 1

        #値が異常になるのを阻止
        curr_sigma = np.clip(curr_sigma, 0.01, 0.996)
        curr_ccov1 = np.clip(curr_ccov1, 0.001, 0.5)
        curr_ccovmu = np.clip(curr_ccovmu, 0.001, 0.5)
        #print("sigma=",curr_sigma)
        #print("ccov=",curr_ccov1)
        #print("ccovmu=",curr_ccovmu)

        #パラメータの決定
        es.sigma = curr_sigma
        es.opts.set({'ccov1':curr_ccov1, 'ccovmu':curr_ccovmu})

        candidates = es.ask()
        fitness = [optimize_K_func(x,params) for x in candidates]

        es.tell (candidates, fitness)

        #---成功時のメモリを保存---
        prev_best = best_fitness #最大値


        suc_sigma = [] #成功メモリ保存用
        suc_ccov1 = []
        suc_ccovmu = []
        delta_E = []
        
        for i, fit in enumerate(fitness):
            if fit < prev_best:
                print("今回の評価値",fit)
                if(prev_best - fit) >= 0.01:
                #---各パラメータ保存---
                    delta_E.append(abs(prev_best - fit))
                    suc_sigma.append(curr_sigma)
                    suc_ccov1.append(curr_ccov1)
                    suc_ccovmu.append(curr_ccovmu)
                    stagnation_counter = 0

                if fit < best_fitness:
                    best_fitness = fit 
                    best_solution = candidates[i]
                    #stagnation_counter = 0
                    archive.append(candidates[i])
        
        if len(suc_sigma) != 0 and len(suc_ccov1) != 0 and len(suc_ccovmu) != 0:
            s_sigma = np.array(suc_sigma)
            s_ccov1 = np.array(suc_ccov1)
            s_ccovmu = np.array(suc_ccovmu)
            w = np.array(delta_E)
            mem_sigma[k] = np.average(s_sigma * s_sigma, weights = w) / np.average(s_sigma, weights = w)
            ave_c1 = np.average(s_ccov1,weights = w)
            ave_cmu = np.average(s_ccovmu,weights = w)
            mem_ccov[k] = (ave_c1,ave_cmu)
            k = k + 1
            if k > (H - 1):
                k = 0
        else:
            stagnation_counter += 1

        print("最高値",best_fitness)

        #print("記録メモリ sigma = ",mem_sigma)
        #print("記録メモリ ccov = ",mem_ccov)
        if best_fitness > -13.0:
            if stagnation_counter > 150:
                print("探索をやり直します")
                #mem_sigma = deque([sigma] * H,maxlen = H)
                #mem_ccov = deque([(init_ccov1,init_ccovmu)] * H, maxlen = H)
                if len(archive) > 0:
                    junp_noise = 0.1
                    base_point = random.choice(list(archive))
                    noise = np.random.normal(0, junp_noise, size = xdim)
                    restart_point = np.clip(base_point + noise, lower_bounds, upper_bounds)
                    new_sigma = np.random.choice(list(mem_sigma))
                    es.result_pretty()
                    es = CMAEvolutionStrategy(restart_point, new_sigma , opts)
                else:
                    #es.sigma = np.random.choice(list(mem_sigma)) * 1.5
                    es.sigma = 0.7
                stagnation_counter = 0
                counter = 0

    return best_solution, best_fitness
    
def optimize_K(
    eta: float,
    number_of_rings: int,
    rng: np.random.Generator,
    params: OptimizeKParams,
    num_start: int = 10
) -> tuple[npt.NDArray[np.float_], float]:

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    bounds = [(1e-12, eta) for _ in range(number_of_rings + 1)]
    bounds_array=np.array(bounds) 
    popsize = 4 + math.floor(3 * math.log(number_of_rings+1)) + 8
    sigma = 0.7
    generations = 500
    num_starts = 1
    initials = [rng.uniform(1e-12, eta, size=(number_of_rings + 1,))
                for _ in range(num_starts)]

    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(SHACMA_run, initial, bounds_array, popsize, sigma, generations, params)
                   for initial in initials]

        results = [f.result() for f in futures]
        print(results)
    print(len(results))

    # 一番良かったやつを選ぶ
    best_solution, best_fitness = min(results, key=lambda x: x[1])
    E: float = -best_fitness
    K: npt.NDArray[np.float_] = best_solution
    return K,E
    
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
       # N = [78,78,78,117,117,117] #6th
        N = [88,88,88,110,110,110]
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
        """
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
        """
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
    # K_minとK_maxを定義

    K_min = 1e-12
    K_max = params.eta
    
    # --- 【代替クリッピング処理の再適用】 ---
    # Kが浮動小数点誤差でわずかに eta_max を超えるのを防ぐ
    K_clamped = np.minimum(K, K_max)
    K_clamped = np.maximum(K_clamped, K_min)
    # ----------------------------------------

    

    x = calculate_x(center_wavelength=params.center_wavelength, FSR=params.FSR)
    y = simulate_transfer_function(
        wavelength=x,
        L=params.L,
        K=K_clamped,
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

