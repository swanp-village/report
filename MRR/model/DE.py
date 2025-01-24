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
import numpy as np






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
   # L:np.array([0.000055,0.000055,0.000055,0.0003297,0.0003297,0.0000824,0.0000824,0.0000824])
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



def optimize_K(
    eta: float,
    number_of_rings: int,
    rng: np.random.Generator,
    params: OptimizeKParams,
) -> tuple[npt.NDArray[np.float_], float]:
    bounds = [(1e-12, eta) for _ in range(number_of_rings + 1)]
    bounds_array=np.array(bounds) 
    #initial=np.random.uniform(1e-12, eta, size=(number_of_rings+1,))
    sampler = LatinHypercube(d=number_of_rings+1)  # 次元数を1に設定
    samples = sampler.random(n=1)  # 1サンプルだけ生成（shape: (1, number_of_rings + 1)）
    initial = samples.flatten() * eta
    #initial = sampler.random(n=number_of_rings + 1).flatten() * eta  # 0からetaの範囲でスケーリング
    popsize = 4 + math.floor(3 * math.log(number_of_rings+1)) + 5
    #sigma = 0.3*(number_of_rings+1)/9
    sigma = 0.3
    generations = (number_of_rings+1) * 100

    
    optimizer=CMA(
        bounds=bounds_array,
        mean=initial,
        sigma=sigma,
        population_size=popsize
    )
    best_solution = None
    best_fitness = float("inf")
    best_genaration = -1

 
    for generation in range(generations):
        solutions = []
        for _ in range(popsize):
            # Ask a parameter
            x=optimizer.ask()
            value = optimize_K_func(x, params)
            solutions.append((x,value))
            if value < best_fitness:
                best_fitness = value
                best_solution = x 
                best_genaration = generations
        optimizer.tell(solutions)

    print(f"Best fitness achieved at generation {best_generation}.")  # 追加: 最適な世代を出力
    E: float = -best_fitness
    K: npt.NDArray[np.float_] = best_solution
    

    return K,E


"""
        fitness_history.append(best_fitness)

        # fitness_historyが10個を超えたら、古い評価値を削除
        if len(fitness_history) > 10:
            fitness_history.pop(0)

        # 過去の評価値を使ってfitness_changeを計算
        if len(fitness_history) > 1:
            fitness_change = abs(fitness_history[-1] - fitness_history[-2])
        else:
            fitness_change = float('inf')  # 最初は比較できないので大きな差を設定

        # stagnation_countを増加させる条件
        if fitness_change < 0.1:
            stagnation_count += 1
        else:
            stagnation_count = 0  # 改善があった場合はリセット

        # stagnation_countが10に達した場合にσを減少
        if stagnation_count >= 30:
            sigma *= 1.05  # σを減少させる（探索範囲を狭める）
            print(f"Generation {generation}: No significant improvement (change < 0.1), decreasing sigma to {sigma}.")

            # 新しいσを反映させるために新たにoptimizerを初期化
            optimizer = CMA(
                bounds=bounds_array,
                mean=optimizer.mean,  # 以前の最良解を引き継ぐ
                sigma=sigma,
                population_size=popsize
            )

            stagnation_count = 0  # カウントをリセット

        # 評価の確認
        print(f"Generation {generation}, Best Fitness: {best_fitness}, Fitness Change: {fitness_change}, Stagnation Count: {stagnation_count}")

        optimizer.tell(solutions)


    E: float = -best_fitness
    K: npt.NDArray[np.float_] = best_solution
    

    return K,E
#　リスタート戦略今後複雑な処理が増えたら使ってみてね
        # 評価値を履歴に追加
        fitness_history.append(best_fitness)
        # 進行状況を表示
        if generation % 50 == 0:
            print(f"Generation {generation}, Best Fitness: {best_fitness}")

        # fitness_historyが10個を超えたら、古い評価値を削除
        if len(fitness_history) > 30:
            fitness_history.pop(0)

        # 過去10世代分の評価値を使ってfitness_changeを計算
        if len(fitness_history) > 1:
            fitness_change = abs(fitness_history[-1] - fitness_history[-2])
        else:
            fitness_change = float('inf')  # 最初は比較できないので大きな差を設定
    
        # stagnation_countを増加させる条件
        if fitness_change < 0.05:
            stagnation_count += 1
        else:
            stagnation_count = 0  # 改善があった場合はリセット

        # stagnation_countが10に達した場合にリスタート
        if stagnation_count >= 100:
            print(f"Generation {generation}: No significant improvement (change < 0.05), restarting optimization.")
            if restart_count >= max_restart:
                print("Maximum restarts reached. Ending optimization")
                break
    
         # リスタート
            restart_count += 1
            sampler = LatinHypercube(d=number_of_rings+1)  # 次元数を1に設定
            samples = sampler.random(n=1)  # 1サンプルだけ生成（shape: (1, number_of_rings + 1)）
            initial = samples.flatten() * eta
            sigma=0.3 
            optimizer = CMA(  
                bounds=bounds_array,
                mean=initial,  # 以前の最良解を引き継ぐ
                sigma=sigma,
                population_size=popsize
            )
            best_solution = None
            best_fitness = float("inf")
            fitness_history.clear()
            stagnation_count = 0  # カウントをリセット

def optimize_K(
    eta: float,
    number_of_rings: int,
    rng: np.random.Generator,
    params: OptimizeKParams,
) -> tuple[npt.NDArray[np.float_], float]:
    bounds = [(1e-12, eta) for _ in range(number_of_rings + 1)]
    best_fitness = float("inf")  # 初期化
    best_generation = -1  # 初期化: 最良値を達成した世代

    def callback(xk, convergence):
        nonlocal best_fitness, best_generation
        current_fitness = -optimize_K_func(xk, params)  # 現在の適応度を計算
        generation = len(callback.generations)
        callback.generations.append((generation, current_fitness))

        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_generation = generation

        print(f"Generation {generation}: Current Best Fitness = {current_fitness}")

    callback.generations = []  # コールバックに世代情報を記録

    result = differential_evolution(
        optimize_K_func,
        bounds,
        args=(params,),
        strategy="currenttobest1bin",
        workers=-1,
        updating="deferred",
        popsize=15,
        maxiter=500,
        seed=rng,
    )
    print(f"Best fitness achieved at generation {best_generation}.")  # 最適な世代を出力
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
        N = [78,78,78,117,117,117] #6th
        #N = [78,78,78,468,468,117,117,117] #8th
        #N = [117,117,117,156,156,156,117,117,156,156] #10th
        #N = [117,117,117,117,468,468,468,78,78,78,78,78] #12th
        #N = [117,117,117,117,468,468,468,468,78,78,78,78,78,78] #14th
        
        L = calculate_ring_length(center_wavelength=center_wavelength, n_eff=n_eff, N=N)
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
        if E > 10:
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
