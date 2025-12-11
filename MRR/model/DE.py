from dataclasses import dataclass
import joblib
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
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple
from scipy.stats import norm
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
        min_fit = min(fitnesses)
        if min_fit < best_fitness:
            print("best_fitness",best_fitness)
            print("min_fitness",min_fit)
            best_fitness = min_fit
            print("new_best",best_fitness)
            best_solution = candidates[fitnesses.index(min_fit)]

        # ログ出力（任意）
        #if generation % 50 == 0 or generation == generations - 1:
            #print(f"Gen {generation}: sigma = {es.sigma:.4f}, best_fitness = {best_fitness:.6f}")


    return best_solution, best_fitness  

# --- 必須: CMA-ESを初期データ収集用として実行するヘルパー関数 ---
def check_overfitting(model, X_train, Y_train, X_test, Y_test):
    """
    訓練データとテストデータに対する R^2 スコアを計算し、過学習を判断する。
    
    Args:
        model: 訓練済みの MLPRegressor モデル
    """
    
    # 訓練データに対する予測
    Y_train_pred = model.predict(X_train)
    # テストデータに対する予測
    Y_test_pred = model.predict(X_test)

    # R^2 スコアの計算
    r2_train = r2_score(Y_train, Y_train_pred)
    r2_test = r2_score(Y_test, Y_test_pred)
    
    print("--- R^2 スコアによる過学習診断 ---")
    print(f"訓練データ R^2: {r2_train:.4f}")
    print(f"テストデータ R^2: {r2_test:.4f}")
    
    if r2_train > 0.99 and r2_test < 0.90:
        print("\n🚨 診断結果: 重大な過学習が発生しています。")
        print("モデルは訓練データのノイズに過剰に適合しています。")
        print("▶︎ 解決策: 'alpha' (正則化) パラメータを増やして再訓練してください。")
    elif r2_train < 0.90:
        print("\n⚠️ 診断結果: モデルがアンダーフィッティング（学習不足）です。")
    else:
        print("\n✅ 診断結果: 汎化性能は良好です。最適化の問題は探索戦略にある可能性があります。")
        
    return r2_train, r2_test
    
def normalize_K(K_physical: np.ndarray, eta_max: float) -> np.ndarray:
    #物理スケール [1e-12, eta] から [0, 1] に正規化する
    K_min = 1e-12
    K_range = eta_max - K_min
    K_normalized = (K_physical - K_min) / K_range
    return np.clip(K_normalized, 0.0, 1.0)

def denormalize_K(K_normalized: np.ndarray, eta_max: float) -> np.ndarray:
    K_min = 1e-12
    K_range = eta_max - K_min
    K_physical = K_normalized * K_range + K_min
    
    # 【代替クリッピング処理】
    # 物理的な上限 eta_max を超えないように制限
    K_physical = np.minimum(K_physical, eta_max) 
    # 下限 K_min (1e-12) より小さくならないように制限
    K_physical = np.maximum(K_physical, K_min)
    
    return K_physical

def get_beta_schedule(iteration: int, max_iterations: int) -> float:

    # 初期値 (探索優先): 50.0 
    beta_start = 2
    
    # 最終値 (活用優先): 10.0
    beta_end = 0.5
    
    # 全体の約80%まで徐々に減少させる
    decay_ratio = 0.8
    decay_iterations = int(max_iterations * decay_ratio)

    if iteration >= decay_iterations:
        # 後半20%は最終値に固定
        beta = beta_end
    else:
        # 線形に減少させる
        beta = beta_start - (beta_start - beta_end) * (iteration / decay_iterations)

    return beta
 
# --- 【ANNアンサンブル予測関数】 ---
def predict_ensemble(K_2d: np.ndarray,ensemble_models: List[MLPRegressor])-> float:
    predictions = np.array([model.predict(K_2d)[0] for model in ensemble_models])
    
    # 予測の平均を mu に、標準偏差を sigma に設定
    mu = np.mean(predictions)
    sigma = np.std(predictions)
    
    return mu, sigma


def visualize_ann_landscape(ensemble_models: List, params: 'OptimizeKParams', N_rings: int):

    
    # --- 1. 設定と初期化 ---
    N_dim = N_rings + 1
    index1, index2 = 0, 1  # 動かす結合率のインデックス (K[0] と K[1] を動かす)
    
    k_min, k_max = 0.0, 1.0  # 正規化された [0, 1] 空間
    resolution = 50 
    
    # 残りの変数の固定値 (全て正規化された 0.5 に固定)
    fixed_K_value_norm = 0.5 
    fixed_K_array_norm = np.full(N_dim, fixed_K_value_norm)
    
    # --- 2. グリッドの生成 ---
    k1_range = np.linspace(k_min, k_max, resolution)
    k2_range = np.linspace(k_min, k_max, resolution)
    
    K1_norm, K2_norm = np.meshgrid(k1_range, k2_range)
    Z_mu = np.zeros(K1_norm.shape)  # 予測平均 (μ) を格納する配列
    
    # --- 3. グリッドの評価（ANN予測） ---
    print(f"評価開始: {resolution * resolution} 回のANN予測を実行中...")
    
    for i in range(resolution):
        for j in range(resolution):
            # 探索点の作成 (K[0], K[1] だけを動かし、残りは固定)
            K_candidate_norm = fixed_K_array_norm.copy()
            K_candidate_norm[index1] = K1_norm[i, j]
            K_candidate_norm[index2] = K2_norm[i, j]
            
            # ANNアンサンブルで予測を実行
            # predict_ensemble は (mu, sigma) のタプルを返す
            mu, sigma = predict_ensemble(K_candidate_norm.reshape(1, -1), ensemble_models)
            
            # 予測平均 (μ) を格納。これが滑らかな探索地形となる。
            Z_mu[i, j] = mu
            
    print("ANN予測完了。")
    
    # --- 4. 3D描画 ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # サーフェスプロットの作成 (cmap='viridis'で滑らかさを強調)
    surf = ax.plot_surface(K1_norm, K2_norm, Z_mu, 
                           cmap='viridis', 
                           edgecolor='none', 
                           alpha=0.8,
                           rstride=1, cstride=1)
    
    # 軸ラベルの設定
    ax.set_xlabel(f'Normalized Coupling K[{index1}]')
    ax.set_ylabel(f'Normalized Coupling K[{index2}]')
    ax.set_zlabel('Predicted Fitness ($\mu$ = F)')
    ax.set_title('ANN Surrogate Model Landscape (Smoothed)')
    
    # カラーバーの追加
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Predicted Fitness')
    
    plt.show()


# 10万点モデルの予測平均を直接返す関数
def acquisition_function_ann(K_candidate, ensemble_models):
    mu, _ = predict_ensemble(K_candidate.reshape(1, -1), ensemble_models)
    return mu # CMA-ESはこの mu を最小化する
    
#R^2計算用bulk
def predict_ensemble_mu_bulk(X_data: np.ndarray, ensemble_models: List[MLPRegressor]) -> np.ndarray:
    """
    複数点（バルク）の入力データ X_data に対し、アンサンブルモデルの予測平均(μ)を計算する。
    """
    # X_data がリストで来る可能性に備え、ここでndarrayに変換（念のため）
    if isinstance(X_data, list):
        X_data = np.array(X_data)
        
    predictions = np.array([model.predict(X_data).flatten() for model in ensemble_models])
    mu = np.mean(predictions, axis=0)
    return mu

FILENAME_PREFIX = "mrr_sao_model"
#FSR=20nm
def save_sao_state(ensemble_models, X_train, Y_train, best_K_norm, best_fitness):
    """ANNアンサンブルモデルとデータをファイルに保存する。"""
    try:
        joblib.dump(ensemble_models, f'{FILENAME_PREFIX}_20_2ensemble.pkl')
        np.save(f'{FILENAME_PREFIX}_20X_2_train.npy', np.array(X_train))
        np.save(f'{FILENAME_PREFIX}_20Y_2_train.npy', np.array(Y_train))
        metadata = {'best_K_norm': best_K_norm, 'best_fitness': best_fitness}
        joblib.dump(metadata, f'{FILENAME_PREFIX}_20_2metadata.pkl')
        print(f"✅ モデルとデータ ({len(X_train)}点) を正常に保存しました。")
    except Exception as e:
        print(f"モデル保存中にエラーが発生しました: {e}")

def load_sao_state():
    """保存されたモデルとデータをファイルから読み込む。"""
    try:
        ensemble_models = joblib.load(f'{FILENAME_PREFIX}_20_2ensemble.pkl')
        X_train = np.load(f'{FILENAME_PREFIX}_20X_2_train.npy').tolist()
        Y_train = np.load(f'{FILENAME_PREFIX}_20Y_2_train.npy').tolist()
        metadata = joblib.load(f'{FILENAME_PREFIX}_20_2metadata.pkl')
        print("✅ 訓練済みモデルとデータを正常に読み込みました。")
        return ensemble_models, X_train, Y_train, metadata['best_K_norm'], metadata['best_fitness'], True
    except FileNotFoundError:
        print("🚨 保存ファイルが見つかりません。新規にSAOを構築します。")
        return None, [], [], None, float("inf"), False
#FSR=35
"""
def save_sao_state(ensemble_models, X_train, Y_train, best_K_norm, best_fitness):
    #ANNアンサンブルモデルとデータをファイルに保存する。
    try:
        joblib.dump(ensemble_models, f'{FILENAME_PREFIX}_35ensemble.pkl')
        np.save(f'{FILENAME_PREFIX}_35X_train.npy', np.array(X_train))
        np.save(f'{FILENAME_PREFIX}_35Y_train.npy', np.array(Y_train))
        metadata = {'best_K_norm': best_K_norm, 'best_fitness': best_fitness}
        joblib.dump(metadata, f'{FILENAME_PREFIX}_35metadata.pkl')
        print(f"✅ モデルとデータ ({len(X_train)}点) を正常に保存しました。")
    except Exception as e:
        print(f"モデル保存中にエラーが発生しました: {e}")

def load_sao_state():
    #保存されたモデルとデータをファイルから読み込む。
    try:
        ensemble_models = joblib.load(f'{FILENAME_PREFIX}_35ensemble.pkl')
        X_train = np.load(f'{FILENAME_PREFIX}_35X_train.npy').tolist()
        Y_train = np.load(f'{FILENAME_PREFIX}_35Y_train.npy').tolist()
        metadata = joblib.load(f'{FILENAME_PREFIX}_35metadata.pkl')
        print("✅ 訓練済みモデルとデータを正常に読み込みました。")
        return ensemble_models, X_train, Y_train, metadata['best_K_norm'], metadata['best_fitness'], True
    except FileNotFoundError:
        print("🚨 保存ファイルが見つかりません。新規にSAOを構築します。")
        return None, [], [], None, float("inf"), False
"""
def optimize_K(
    eta: float,
    number_of_rings: int,
    rng: np.random.Generator,
    params: OptimizeKParams,
    build_model_only: bool = False,
) -> tuple[npt.NDArray[np.float_], float]:
    #-----初期設定-----
    N_dim = number_of_rings + 1
    ensemble_models, X_train, Y_train, best_K_norm, best_fitness, loaded = load_sao_state()
    bounds_normalized = np.array([(0.0, 1.0) for _ in range(N_dim)])

    if not loaded:
        hidden_layer_sizes = (512,256,128,128) 
        NUM_ENSEMBLE = 1
        base_ann_model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes, 
            max_iter=30000, 
            learning_rate_init = 0.0005,
            activation='relu', 
            solver='adam', 
            random_state=42,
            verbose = True,
            n_iter_no_change = 100
        )
        ensemble_models = [clone(base_ann_model) for _ in range(NUM_ENSEMBLE)]
    #変数
        initial_samples = 50000 # 凹凸対策として10Nに増やす
    #データセット
        X_train = []
        Y_train = []
        best_K_norm: Optional[np.ndarray] = None
        best_fitness = float("inf")
        
    
    #-----データ収集-----
        print("データ収集開始")
        lhs = LatinHypercube(d=N_dim, seed=rng)
        initial_K_samples = lhs.random(n = initial_samples)
        for K_sample in initial_K_samples:
        #評価関数で計算
            K_sample_phy = denormalize_K(K_sample,params.eta)
            train_fitness = optimize_K_func(K_sample_phy,params)
            X_train.append(K_sample)
            Y_train.append(train_fitness)

            if train_fitness < best_fitness:
                best_fitness = train_fitness
                best_K_norm = K_sample
        # データ収集ループの直後に追加
            Y_arr_initial = np.array(Y_train)

        print(f"最大値 (最良の解): {Y_arr_initial.min():.6f}")
    X_full_arr = np.array(X_train)
    Y_full_arr = np.array(Y_train).ravel() # Y_trainは平坦化

    # 全データ (X_train, Y_train) を訓練用とテスト用に分割
    # X_train_split: 訓練に使用するデータ (90%)
    # X_test: 診断に使用するデータ (10%)
    X_train_split, X_test, Y_train_split, Y_test = train_test_split(
        X_full_arr, 
        Y_full_arr, 
        test_size=0.1, 
        random_state=42 # 再現性の確保
    )
    print(f"データセットを分割しました: 訓練点数={len(X_train_split)}, テスト点数={len(X_test)}")
    
    
    # 🚨 【修正 2】: モデル訓練に分割後の訓練データを使用
    # X_arr = np.array(X_train)  <-- 元々この行があった場合、削除/置換
    # Y_arr = np.array(Y_train)  <-- 元々この行があった場合、削除/置換
    X_arr = X_train_split # 分割後の訓練データ
    Y_arr = Y_train_split # 分割後の訓練データ
    #for iteration in range (MAX_SAO_ITERATIONS):
        #current_beta = get_beta_schedule(iteration, MAX_SAO_ITERATIONS)
        #X_arr_1 = np.array(X_train)
        #Y_arr_1 = np.array(Y_train)
    for model in ensemble_models:
            model.fit(X_arr,Y_arr.ravel())
    save_sao_state(ensemble_models, X_train, Y_train, best_K_norm, best_fitness)
    print(f"STEP 3: SAOモデル訓練完了。")
    if build_model_only:
            # モデル構築のみを目的とする場合、ここで終了
        return denormalize_K(best_K_norm, eta), -best_fitness
    
    
    Y_train_pred = predict_ensemble_mu_bulk(X_train_split, ensemble_models)
    Y_test_pred = predict_ensemble_mu_bulk(X_test, ensemble_models)

    r2_train = r2_score(Y_train_split, Y_train_pred)
    r2_test = r2_score(Y_test, Y_test_pred)
    
    # check_overfittingの診断ロジックを直接ここに組み込む
    print("--- R^2 スコアによる過学習診断 ---")
    print(f"訓練データ R^2: {r2_train:.4f}")
    print(f"テストデータ R^2: {r2_test:.4f}")
    
    if r2_train > 0.99 and r2_test < 0.90:
        print("\n🚨 診断結果: 重大な過学習が発生しています。")
        print("▶︎ 解決策: 'alpha' (正則化) パラメータを増やして再訓練してください。")
    elif r2_train < 0.90:
        print("\n⚠️ 診断結果: モデルがアンダーフィッティング（学習不足）です。")
    else:
        print("\n✅ 診断結果: 汎化性能は良好です。最適化の問題は探索戦略にある可能性があります。") 
    #-----獲得関数の最適化-----
    if not build_model_only:
        
        def final_optimization_wrapper(K_candidate):
            return acquisition_function_ann(K_candidate, ensemble_models)
        
    # 最適化の初期スタート点: 初期データで見つけた最良点からスタート
        initial_start_norm = best_K_norm if best_K_norm is not None else np.full(N_dim, 0.5)
    
    # CMA-ESの最終実行 (獲得関数なし、直接μを最小化)
        final_K_norm, final_fitness = cma_run(
            initial=initial_start_norm, 
            bounds_array=bounds_normalized,
            popsize=4 + math.floor(3 * math.log(number_of_rings+1)) + 8,
            sigma=0.3, # モデルベースでは0.3~0.5程度の安定した値で良い
            generations=500, # 収束するまで十分な世代数を確保
            params=params,
            objective_func=final_optimization_wrapper 
        )
    
    #-----【最終検証】-----
    # CMA-ESが見つけた最適解を、最後に一度だけ真の評価関数で検証する
        final_K_phy = denormalize_K(final_K_norm, params.eta)
        true_final_fitness = optimize_K_func(final_K_phy, params)
    
        print(f"最終検証: CMA-ES予測={final_fitness:.6f}, 真の評価値={true_final_fitness:.6f}")
    
    # --- 可視化は残すが、元のコードには含まれていないため関数呼び出しのみ残す ---
        visualize_ann_landscape(ensemble_models, params, number_of_rings)
        #r2_train, r2_test = check_overfitting(ensemble_models, X_train, Y_train, X_test, Y_test)
        
    
    # ----- [最終結果] -----
        E: float = -true_final_fitness
        K: npt.NDArray[np.float_] = final_K_phy # 非正規化されたKを返す

        return K, E
    """
        def acquisition_wrapper(K_candidate):
            return acquisition_function_ann(K_candidate, ensemble_models)
        lower_bounds = bounds_normalized[:, 0]
        upper_bounds = bounds_normalized[:, 1]

        if best_K_norm is None or rng.uniform(0, 1) < 0.9: 
    # データ収集がまだ成功していない場合、または探索を強制する場合
            lower_bounds = bounds_normalized[:, 0]
            upper_bounds = bounds_normalized[:, 1]
            initial_start_norm = rng.uniform(low=lower_bounds, high=upper_bounds, size=(N_dim,))
        else:
    # 最良解の近傍からスタートする (活用)
            initial_start_norm = best_K_norm
        acq_best_K, _ = cma_run(
            initial=initial_start_norm, 
            bounds_array=bounds_normalized,
            popsize=4 + math.floor(3 * math.log(number_of_rings+1)) + 8, 
            sigma=1.0, 
            generations=200, 
            params=params,
            objective_func=acquisition_wrapper 
        )
        
    #-----真値の再評価とデータの更新-----
        # 獲得関数が提案した点 (acq_best_K) を元の評価関数で確認
        print(acq_best_K)
        acq_best_K_phy = denormalize_K(acq_best_K,params.eta)
        true_fitness_new = optimize_K_func(acq_best_K_phy, params)
        
        
        # データセットを更新
        X_train.append(acq_best_K)
        Y_train.append(true_fitness_new)
        print(f"今回の評価値",true_fitness_new)
        #print(best_fitness)
        # 全体の最良解を更新
        if true_fitness_new < best_fitness:
            best_fitness = true_fitness_new
            best_K_norm = acq_best_K
            
        print(f"STEP 4: 真値再評価完了。Best Fitness (True) = {best_fitness:.6f}")

    print(">>> 3D SAOモデルの地形を可視化中...")
            # 訓練済みのモデルを可視化関数に渡す
    visualize_ann_landscape(ensemble_models, params, number_of_rings)
    # ----- [最終結果] -----
    E: float = -best_fitness
    K: npt.NDArray[np.float_] = denormalize_K(best_K_norm if best_K_norm is not None else np.zeros(N_dim), params.eta)

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
       # N = [88,88,110,110,110,110]
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

