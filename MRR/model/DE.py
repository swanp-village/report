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


#CMA-ES動作コード_CMA
def cma_run(initial, bounds_array, popsize, sigma, generations, params):
    optimizer=CMA(
            bounds=bounds_array,
            mean=initial,
            sigma=sigma,
            population_size=popsize
        )
    best_solution = None
    best_fitness = float("inf")
    for generation in range(generations):
        solutions = []
        for _ in range(popsize):
            # Ask a parameter
            x=optimizer.ask()
            value = optimize_K_func(x, params)
            solutions.append((x,value))
            if value < best_fitness :
                best_fitness = value
                best_solution = x 
       
        optimizer.tell(solutions)
            
    return best_solution,best_fitness
"""
#CMA-ES動作コード_pycma
def cma_run(initial, bounds_array, popsize, sigma, generations, params):
    # bounds_array: shape (N, 2)
    lower_bounds = bounds_array[:, 0]
    upper_bounds = bounds_array[:, 1]

    opts = {
        'bounds': [lower_bounds, upper_bounds],
        'popsize': popsize,
        'verb_log': 0,
        'verbose': -9,  # suppress internal logs
        'tolfun':1e-13,
    }

    es = CMAEvolutionStrategy(initial, sigma, opts)

    best_solution = None
    best_fitness = float("inf")

    for generation in range(generations):
        candidates = es.ask()
        fitnesses = [optimize_K_func(x, params) for x in candidates]
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

        if es.stop():
            print(f"Optimization stopped at generation {generation}.")
            print(f"Stop conditions met: {es.stop()}")
            break # ループを抜ける


    return best_solution, best_fitness

def optimize_K(
    eta: float,
    number_of_rings: int,
    rng: np.random.Generator,
    params: OptimizeKParams,
    num_start: int = 6
) -> tuple[npt.NDArray[np.float_], float]:
    bounds = [(1e-12, eta) for _ in range(number_of_rings + 1)]
    bounds_array=np.array(bounds) 
    popsize = 4 + math.floor(3 * math.log(number_of_rings+1)) + 8
    sigma = 0.7
    generations = 500
    num_starts = 6
    initials = [rng.uniform(1e-12, eta, size=(number_of_rings + 1,))
                for _ in range(num_starts)]

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(cma_run, initial, bounds_array, popsize, sigma, generations, params)
                   for initial in initials]

        results = [f.result() for f in futures]
        print(results)

    # 一番良かったやつを選ぶ
    best_solution, best_fitness = min(results, key=lambda x: x[1])
    E: float = -best_fitness
    K: npt.NDArray[np.float_] = best_solution
    return K,E

"""
#一般設計
def optimize_K(
    eta: float,
    number_of_rings: int,
    rng: np.random.Generator,
    params: OptimizeKParams,
    num_start: int = 6
) -> tuple[npt.NDArray[np.float_], float]:
    bounds = [(1e-12, eta) for _ in range(number_of_rings + 1)]
    bounds_array=np.array(bounds) 
    initial=np.random.uniform(1e-12, eta, size=(number_of_rings+1,))
    popsize = 4 + math.floor(3 * math.log(number_of_rings+1)) + 8
    sigma = 0.3
    #sigma = 0.07 #誤差検討用σ
    generations = 500




    optimizer=CMA(
        bounds=bounds_array,
        mean=initial,
        sigma=sigma,
        population_size=popsize
    )
    best_solution = None
    best_fitness = float("inf")
    for generation in range(generations):
        solutions = []
        for _ in range(popsize):
            # Ask a parameter
            x=optimizer.ask()
            value = optimize_K_func(x, params)
            solutions.append((x,value))
            if value < best_fitness :
                best_fitness = value
                best_solution = x 
       
        optimizer.tell(solutions)
        print(best_fitness)
        
        
    E: float = -best_fitness
    K: npt.NDArray[np.float_] = best_solution
    

    return K,E

#差分進化法
def optimize_K(
    eta: float,
    number_of_rings: int,
    rng: np.random.Generator,
    params: OptimizeKParams,
) -> tuple[npt.NDArray[np.float_], float]:
    bounds = [(1e-12, eta) for _ in range(number_of_rings + 1)]

    result = differential_evolution(
        combined_evaluation,
        bounds,
        args=(params,),
        strategy="currenttobest1bin",
        workers=-1,
        updating="deferred",
        popsize=15,
        maxiter=500,
        seed=rng,
    )
    E: float = -result.fun
    K: npt.NDArray[np.float_] = result.x

    return K, E
"""

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

