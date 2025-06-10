import cvxpy as cp
import numpy as np
import os
import pandas as pd
from src.tasks.pre_processing.settings import Settings
from src.utils import ErrorHandler, ErrorCode, ErrorLevel
from src.helpers.xplor_maker import make_xplor


def load_previous_result(filepath: str) -> np.ndarray | None:
    """
    前回の計算結果をCSVファイルから読み込む
    
    Args:
        filepath: CSVファイルのパス
        
    Returns:
        読み込んだ行列（numpy配列）、失敗時はNone
    """
    try:
        if not os.path.exists(filepath):
            return None
            
        # CSVファイルを読み込み
        df = pd.read_csv(filepath, header=None)
        matrix = df.values
        
        # 正方行列かチェック
        if matrix.shape[0] != matrix.shape[1]:
            print(f"警告: 読み込んだ行列が正方行列ではありません。形状: {matrix.shape}")
            return None
            
        # 数値型に変換
        matrix = matrix.astype(np.float64)
        
        # NaNや無限大の値をチェック
        if not np.isfinite(matrix).all():
            print("警告: 読み込んだ行列に無効な値（NaN/Inf）が含まれています")
            return None
            
        print(f"前回の結果を読み込みました: {filepath}, 形状: {matrix.shape}")
        return matrix
        
    except pd.errors.EmptyDataError:
        print(f"エラー: CSVファイルが空です: {filepath}")
        return None
    except pd.errors.ParserError as e:
        print(f"エラー: CSVファイルの解析に失敗しました: {filepath}, {e}")
        return None
    except Exception as e:
        print(f"エラー: ファイル読み込み中に予期しないエラーが発生しました: {filepath}, {e}")
        return None


def improve_objective_scaling(target_data: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """
    目的関数のスケーリングを改善するためのスケール因子を計算
    
    Args:
        target_data: ターゲットデータ
        weights: 重み行列
        
    Returns:
        tuple[float, float]: (data_scale, weight_scale)
    """
    # データの典型的なスケールを計算
    data_scale = np.sqrt(np.mean(target_data ** 2))
    
    # 重みの典型的なスケールを計算
    weight_scale = np.sqrt(np.mean(weights))
    
    # スケールが小さすぎる場合の調整
    if data_scale < 1e-6:
        data_scale = 1e-3
    if weight_scale < 1e-6:
        weight_scale = 1.0
        
    print(f"Data scale: {data_scale:.2e}, Weight scale: {weight_scale:.2e}")
    return data_scale, weight_scale


def main_loop_cvxpy(
    basis: np.ndarray, 
    target_data: np.ndarray, 
    settings: Settings,
    weights: np.ndarray = None,
    regularization_lambda: float = 0.0,
    initial_P: np.ndarray = None,
    initial_method: str = "identity",  # "identity", "random", "file"
    initial_scale: float = 1.0,
    target_mse: float = 1e-3,  # 目標MSE（これ以下なら十分）
    target_relative_error: float = 5e-2,  # 目標相対誤差（これ以下なら十分）
    zero_constraints: list[tuple[int, int]] = None  # ゼロ制約: [(i,j), ...] のリスト
) -> None:
    """
    CVXPYを用いて半正定値制約付きの最適化問題を解く
    
    Args:
        basis: n×n×grid_x×grid_y×grid_z の基底関数配列
        target_data: grid_x×grid_y×grid_z のターゲットデータ
        settings: 設定オブジェクト
        weights: 重み行列（デフォルトは全て1）
        regularization_lambda: 正則化パラメータ（デフォルトは0）
        initial_P: 初期値行列
        initial_method: 初期値の生成方法 ("identity", "random", "file")
        initial_scale: 初期値のスケール
        target_mse: 目標MSE（これ以下なら十分）
        target_relative_error: 目標相対誤差（これ以下なら十分）
        zero_constraints: 0に固定する成分のリスト [(i,j), ...] 
                         例: [(0,3), (3,0), (1,4), (4,1)] で特定の混成を禁止
    
    Returns:
        None（結果はファイルに保存される）
    """
    error_handler = ErrorHandler()
    
    try:
        # 入力検証
        if basis.ndim != 5:
            error = error_handler.handle(
                f"basisは5次元配列である必要があります。現在の次元: {basis.ndim}",
                ErrorCode.INVALID_INPUT,
                ErrorLevel.CRITICAL
            )
            if error:
                raise error
                
        if target_data.ndim != 3:
            error = error_handler.handle(
                f"target_dataは3次元配列である必要があります。現在の次元: {target_data.ndim}",
                ErrorCode.INVALID_INPUT,
                ErrorLevel.CRITICAL
            )
            if error:
                raise error
        
        n = basis.shape[0]
        grid_shape = target_data.shape
        
        if basis.shape[1] != n:
            error = error_handler.handle(
                "basisの最初の2次元は同じサイズである必要があります",
                ErrorCode.INVALID_INPUT,
                ErrorLevel.CRITICAL
            )
            if error:
                raise error
                
        if basis.shape[2:] != grid_shape:
            error = error_handler.handle(
                f"basisのgrid部分とtarget_dataの形状が一致しません。basis: {basis.shape[2:]}, target: {grid_shape}",
                ErrorCode.INVALID_INPUT,
                ErrorLevel.CRITICAL
            )
            if error:
                raise error
        
        # 重み行列のデフォルト設定
        if weights is None:
            weights = np.ones(grid_shape)
        elif weights.shape != grid_shape:
            error_handler.handle(
                f"重み行列の形状がtarget_dataと一致しません。weights: {weights.shape}, target: {grid_shape}",
                ErrorCode.INVALID_INPUT,
                ErrorLevel.WARNING
            )
            weights = np.ones(grid_shape)
        
        print(f"最適化開始: n={n}, grid_shape={grid_shape}")
        
        # ゼロ制約の検証とログ出力
        if zero_constraints is not None:
            print(f"ゼロ制約が指定されました: {len(zero_constraints)}個の成分")
            for i, (row, col) in enumerate(zero_constraints):
                if not (0 <= row < n and 0 <= col < n):
                    error = error_handler.handle(
                        f"ゼロ制約のインデックスが範囲外です: ({row}, {col}), 有効範囲: [0, {n-1}]",
                        ErrorCode.INVALID_INPUT,
                        ErrorLevel.CRITICAL
                    )
                    if error:
                        raise error
                print(f"  制約 {i+1}: P[{row}, {col}] = 0")
                
                # エルミート性の確保：P[i,j] = 0 なら P[j,i]* = 0 も自動追加
                if row != col and (col, row) not in zero_constraints:
                    zero_constraints.append((col, row))
                    print(f"  エルミート性により追加: P[{col}, {row}] = 0")
        
        # CVXPYの変数定義（n×n の複素エルミート半正定値行列P）
        P = cp.Variable((n, n), hermitian=True)
        
        # 初期値の設定
        if initial_P is None or initial_P.shape != (n, n):
            if initial_method == "identity":
                initial_P = initial_scale * np.eye(n, dtype=complex)
            elif initial_method == "random":
                # 複素数の低ランク行列を生成してエルミート化
                A_real = np.random.randn(n, min(n, 4))
                A_imag = np.random.randn(n, min(n, 4))
                A = A_real + 1j * A_imag
                initial_P = initial_scale * (A @ A.conj().T)
            elif initial_method == "file":
                initial_P = load_previous_result("output/matrix.csv")
                if initial_P is None:
                    initial_P = np.eye(n, dtype=complex)
                    error_handler.handle(
                        "前回の結果ファイルが読み込めません。単位行列を使用します",
                        ErrorCode.NOT_FOUND,
                        ErrorLevel.WARNING
                    )
                else:
                    # ファイルから読み込んだ実数行列を複素数に変換
                    initial_P = initial_P.astype(complex)
        
        # 初期値の検証と設定
        if initial_P is not None:
            # 形状チェック
            if initial_P.shape != (n, n):
                error_handler.handle(
                    f"初期値の形状が不正です。expected: ({n}, {n}), got: {initial_P.shape}",
                    ErrorCode.INVALID_INPUT,
                    ErrorLevel.WARNING
                )
                initial_P = np.eye(n, dtype=complex)
            
            # エルミート性の確保
            initial_P = (initial_P + initial_P.conj().T) / 2
            
            # 半正定値性の確保
            eigenvals, eigenvecs = np.linalg.eigh(initial_P)
            eigenvals_clipped = np.maximum(eigenvals, 1e-10)
            initial_P = eigenvecs @ np.diag(eigenvals_clipped) @ eigenvecs.conj().T
            
            # 初期値を設定
            P.value = initial_P
            print(f"初期値設定完了:")
            print(f"  - トレース: {np.trace(initial_P):.6f}")
            print(f"  - 最小固有値: {np.min(eigenvals_clipped):.2e}")
            print(f"  - フロベニウスノルム: {np.linalg.norm(initial_P, 'fro'):.6f}")
            print(f"  - 複素数対応: True")
        
        # 目的関数のスケーリング改善
        data_scale, weight_scale = improve_objective_scaling(target_data, weights)
        
        # スケーリングされた重みとターゲットデータ
        scaled_weights = weights / weight_scale
        scaled_target = target_data / data_scale
        
        # 差分計算用の変数
        rho_diff = cp.Variable(grid_shape)
        
        # 制約条件: rho_diff = sum_{i,j} P[i,j] * basis[i,j,:,:,:] - scaled_target
        constraints = []
        
        # rho_diffの計算制約を追加（スケーリング考慮）
        rho_expr = cp.Constant(np.zeros(grid_shape))
        for i in range(n):
            for j in range(n):
                # basisもデータスケールで正規化
                # 複素数行列の場合、実部のみを使用（密度は実数のため）
                basis_scaled = basis[i, j, :, :, :] / data_scale
                if np.iscomplexobj(basis_scaled):
                    rho_expr += cp.real(P[i, j] * cp.Constant(basis_scaled))
                else:
                    rho_expr += cp.real(P[i, j]) * cp.Constant(basis_scaled)
        
        constraints.append(rho_diff == rho_expr - cp.Constant(scaled_target))
        
        # 半正定値制約
        constraints.append(P >> 0)  # P は半正定値
        
        # ゼロ制約の追加
        if zero_constraints is not None:
            for row, col in zero_constraints:
                constraints.append(P[row, col] == 0)
            print(f"ゼロ制約を追加しました: {len(set(zero_constraints))}個のユニークな成分")
        
        # 目的関数: スケーリングされた重み付き二乗誤差 + 正則化項
        weighted_diff = cp.multiply(cp.Constant(np.sqrt(scaled_weights)), rho_diff)
        
        # スケール調整された目的関数（値を適度な範囲に保つ）
        objective_scale = max(1.0, 1.0 / (data_scale * weight_scale))
        objective = objective_scale * cp.sum_squares(weighted_diff) / cp.Constant(np.prod(grid_shape))
        
        if regularization_lambda > 0:
            # 正則化項もスケーリングを考慮
            reg_scale = regularization_lambda * (data_scale ** 2)
            objective += reg_scale * cp.sum_squares(P)
        
        print(f"Objective scaling factor: {objective_scale:.2e}")
        print(f"Expected objective magnitude: O({objective_scale * data_scale * weight_scale:.1e})")
        
        # 問題の定義
        problem = cp.Problem(cp.Minimize(objective), constraints)
        
        # 最適化の実行
        print("CVXPY最適化を実行中...")
        try:
            # まずMOSEKを試す（厳しい収束条件）
            problem.solve(
                solver=cp.MOSEK, 
                verbose=True,
                mosek_params={
                    'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-10,
                    'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-10,
                    'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-10,
                    'MSK_DPAR_INTPNT_CO_TOL_MU_RED': 1e-12
                }
            )
        except:
            try:
                # MOSEKが失敗した場合はSCSを試す（厳しい収束条件）
                error_handler.handle(
                    "MOSEKソルバーが利用できません。SCSソルバーを使用します",
                    ErrorCode.NOT_FOUND,
                    ErrorLevel.WARNING
                )
                problem.solve(
                    solver=cp.SCS, 
                    verbose=True,
                    eps_abs=1e-8,      # 実用的な絶対許容誤差（MSE 1e-4に対応）
                    eps_rel=1e-8,      # 実用的な相対許容誤差
                    max_iters=3000,    # 十分な反復数を確保
                    alpha=1.5,         # Douglas-Rachford緩和パラメータ（デフォルト）
                    scale=0.01,        # 双対スケール因子（小さい目的関数値に適応）
                    adaptive_scale=True, # 適応的スケーリングを有効化
                    normalize=True,    # データの前処理正規化を有効化
                    acceleration_lookback=10,  # Anderson加速のメモリを適度に設定
                    # 実用的収束のための追加設定
                    time_limit_secs=300.0,  # 5分でタイムアウト
                    use_indirect=False  # 直接法を使用（より安定）
                )
            except Exception as e:
                error = error_handler.handle(
                    f"最適化に失敗しました: {str(e)}",
                    ErrorCode.VALIDATION,
                    ErrorLevel.CRITICAL
                )
                if error:
                    raise error
        
        # 結果の確認
        if problem.status not in ["infeasible", "unbounded"]:
            if problem.status != "optimal":
                error_handler.handle(
                    f"最適化が最適解に収束しませんでした。ステータス: {problem.status}",
                    ErrorCode.VALIDATION,
                    ErrorLevel.WARNING
                )
            
            # 結果の取得
            P_optimal = P.value
            if P_optimal is None:
                error = error_handler.handle(
                    "最適化結果が取得できませんでした",
                    ErrorCode.VALIDATION,
                    ErrorLevel.CRITICAL
                )
                if error:
                    raise error
            
            print(f"最適化完了。目的関数値: {objective.value}")
            
            # rhoの計算（スケーリングを考慮した逆変換）
            rho_output = np.zeros(grid_shape)
            for i in range(n):
                for j in range(n):
                    # 元のスケールに戻す
                    rho_output += (P_optimal[i, j] * basis[i, j, :, :, :]).real
            
            # 出力ディレクトリの作成
            os.makedirs("output", exist_ok=True)
            
            # rho_output.xplorの保存
            make_xplor(rho_output, "output/rho_output.xplor", "rho_output", settings)
            print("rho_output.xplorを保存しました")
            
            # 複素数行列の保存（実部と虚部を分けて保存）
            P_real = np.real(P_optimal)
            P_imag = np.imag(P_optimal)
            
            # 実部を保存
            P_real_df = pd.DataFrame(P_real)
            P_real_df.to_csv("output/matrix_real.csv", index=False)
            print("matrix_real.csvを保存しました")
            
            # 虚部を保存（ゼロでない場合のみ）
            if np.max(np.abs(P_imag)) > 1e-12:
                P_imag_df = pd.DataFrame(P_imag)
                P_imag_df.to_csv("output/matrix_imag.csv", index=False)
                print("matrix_imag.csvを保存しました")
            else:
                print("虚部はゼロのため、matrix_imag.csvは保存されませんでした")
            
            # 後方互換性のため、実部のみをmatrix.csvとしても保存
            P_real_df.to_csv("output/matrix.csv", index=False)
            print("後方互換性のため、実部をmatrix.csvとしても保存しました")
            
            # 統計情報の出力（元のスケールで評価）
            rms_error = np.sqrt(np.mean((rho_output - target_data) ** 2))
            relative_error = rms_error / np.sqrt(np.mean(target_data ** 2))
            print(f"RMS誤差: {rms_error:.6e}")
            print(f"相対RMS誤差: {relative_error:.6e}")
            
            # スケーリングされた誤差も表示
            scaled_residual = np.sqrt(np.mean(((rho_output / data_scale) - scaled_target) ** 2))
            print(f"スケーリング後のRMS誤差: {scaled_residual:.6e}")
            
            # 収束診断情報の追加
            if hasattr(problem, 'solver_stats') and problem.solver_stats is not None:
                stats = problem.solver_stats
                print(f"\n=== 収束診断 ===")
                print(f"ソルバー: {stats.solver_name if hasattr(stats, 'solver_name') else 'SCS'}")
                print(f"反復回数: {stats.num_iters if hasattr(stats, 'num_iters') else 'N/A'}")
                print(f"解法時間: {stats.solve_time:.3f}秒" if hasattr(stats, 'solve_time') else "解法時間: N/A")
                
                # SCS特有の情報
                if hasattr(stats, 'extra_stats') and stats.extra_stats:
                    extra = stats.extra_stats
                    if 'residual_norm' in extra:
                        print(f"最終残差ノルム: {extra['residual_norm']:.2e}")
                    if 'gap' in extra:
                        print(f"双対ギャップ: {extra['gap']:.2e}")
            
            # 行列Pの性質を確認（複素エルミート行列対応）
            eigenvalues = np.linalg.eigvals(P_optimal)
            # エルミート行列の固有値は実数なので実部を取得
            eigenvalues_real = np.real(eigenvalues)
            eigenvalues_sorted = np.sort(eigenvalues_real)[::-1]
            min_eigenvalue = np.min(eigenvalues_real)
            condition_number = np.max(eigenvalues_real) / max(min_eigenvalue, 1e-12)
            
            print(f"\n=== 複素エルミート行列P の性質 ===")
            print(f"最小固有値: {min_eigenvalue:.2e}")
            print(f"最大固有値: {np.max(eigenvalues_real):.2e}")
            print(f"条件数: {condition_number:.2e}")
            print(f"有効ランク (閾値1e-8): {np.sum(eigenvalues_real > 1e-8)}")
            print(f"上位5固有値: {eigenvalues_sorted[:5]}")
            
            # エルミート性の確認
            hermitian_error = np.max(np.abs(P_optimal - P_optimal.conj().T))
            print(f"エルミート性誤差: {hermitian_error:.2e}")
            
            # PSD制約の評価（数値許容誤差を考慮）
            psd_tolerance = 1e-6  # 数値誤差の許容範囲
            if min_eigenvalue < -psd_tolerance:
                error_handler.handle(
                    f"結果の行列が半正定値ではありません。最小固有値: {min_eigenvalue}",
                    ErrorCode.VALIDATION,
                    ErrorLevel.WARNING
                )
                
                # PSD修正の提案
                print(f"\n🔧 PSD修正オプション:")
                print(f"   - 負の固有値を0にクリップ")
                print(f"   - 正則化パラメータを増加 (現在: {regularization_lambda})")
                
                # 自動PSD修正（複素エルミート行列対応）
                if abs(min_eigenvalue) < 1e-3:  # 小さい負の固有値なら自動修正
                    eigenvalues_corrected = np.maximum(eigenvalues_real, 0)
                    eigenvecs = np.linalg.eigh(P_optimal)[1]
                    P_corrected = eigenvecs @ np.diag(eigenvalues_corrected) @ eigenvecs.conj().T
                    
                    # 修正後の保存（実部と虚部を分けて保存）
                    P_corrected_real = np.real(P_corrected)
                    P_corrected_imag = np.imag(P_corrected)
                    
                    # 実部を保存
                    P_corrected_real_df = pd.DataFrame(P_corrected_real)
                    P_corrected_real_df.to_csv("output/matrix_psd_corrected_real.csv", index=False)
                    
                    # 虚部を保存（ゼロでない場合のみ）
                    if np.max(np.abs(P_corrected_imag)) > 1e-12:
                        P_corrected_imag_df = pd.DataFrame(P_corrected_imag)
                        P_corrected_imag_df.to_csv("output/matrix_psd_corrected_imag.csv", index=False)
                        print(f"   ✓ PSD修正済み行列（実部）を matrix_psd_corrected_real.csv に保存")
                        print(f"   ✓ PSD修正済み行列（虚部）を matrix_psd_corrected_imag.csv に保存")
                    else:
                        print(f"   ✓ PSD修正済み行列（実部のみ）を matrix_psd_corrected_real.csv に保存")
                    
                    print(f"   修正後の最小固有値: {np.min(eigenvalues_corrected):.2e}")
            else:
                print(f"✓ 行列は半正定値です（許容誤差 {psd_tolerance:.1e} 内）")
            
            # 収束に関する追加のアドバイス
            if relative_error > target_relative_error:
                print(f"\n⚠️  相対誤差が目標値を超えています（{relative_error:.1e} > {target_relative_error:.1e}）")
                print("   推奨対策:")
                print(f"   - 正則化パラメータの調整: {regularization_lambda} → {regularization_lambda * 10}")
                print(f"   - より多くの反復回数: {2000} → {4000}")
                print(f"   - より厳しい許容誤差: eps_abs={1e-4} → {1e-5}")
            elif rms_error < target_mse:
                print(f"\n✓ MSE目標達成: {rms_error:.1e} < {target_mse:.1e}")
                print("   実用的には十分な精度です。")
            
            # 実用性の総合判定
            is_practically_converged = (
                rms_error < target_mse and 
                relative_error < target_relative_error and
                abs(min_eigenvalue) < 1e-3
            )
            
            if is_practically_converged:
                print(f"\n🎯 実用的収束達成!")
                print(f"   - MSE: {rms_error:.1e} < {target_mse:.1e} ✓")
                print(f"   - 相対誤差: {relative_error:.1e} < {target_relative_error:.1e} ✓") 
                print(f"   - PSD制約: |最小固有値| = {abs(min_eigenvalue):.1e} < 1e-3 ✓")
            else:
                print(f"\n⚠️  実用的収束未達成。さらなる調整が必要です。")
            
        else:
            error = error_handler.handle(
                f"最適化問題が解けませんでした。ステータス: {problem.status}",
                ErrorCode.VALIDATION,
                ErrorLevel.CRITICAL
            )
            if error:
                raise error
                
    except Exception as e:
        error = error_handler.handle(
            f"予期しないエラーが発生しました: {str(e)}",
            ErrorCode.VALIDATION,
            ErrorLevel.CRITICAL
        )
        if error:
            raise error


def run_practical_optimization(
    basis: np.ndarray, 
    target_data: np.ndarray, 
    settings: Settings,
    **kwargs
) -> dict:
    """
    実用的なSDP最適化の実行
    
    MSE 1e-3以下、相対誤差5%以下を目標とした実用的な設定で最適化を実行
    
    Args:
        basis: 基底関数配列
        target_data: ターゲットデータ
        settings: 設定オブジェクト
        **kwargs: main_loop_cvxpyに渡される追加引数
                 例: zero_constraints=[(0,3), (3,0)] で特定成分を0に固定
    
    Returns:
        dict: 最適化結果の要約
    """
    print("=== 実用的SDP最適化の実行 ===")
    print("目標: MSE < 1e-3, 相対誤差 < 5%")
    
    # デフォルト設定
    default_kwargs = {
        'regularization_lambda': 1e-6,
        'target_mse': 1e-3,
        'target_relative_error': 0.05,
        'initial_method': 'identity'
    }
    default_kwargs.update(kwargs)
    
    try:
        main_loop_cvxpy(basis, target_data, settings, **default_kwargs)
        
        # 結果の読み込みと評価（複素数対応）
        if os.path.exists("output/matrix_real.csv"):
            P_real = pd.read_csv("output/matrix_real.csv", header=None).values
            
            # 虚部の読み込み（存在する場合）
            if os.path.exists("output/matrix_imag.csv"):
                P_imag = pd.read_csv("output/matrix_imag.csv", header=None).values
                P_result = P_real + 1j * P_imag
                print("複素数行列として結果を読み込みました")
            else:
                P_result = P_real.astype(complex)
                print("実数行列として結果を読み込みました")
            
            eigenvals = np.linalg.eigvals(P_result)
            eigenvals_real = np.real(eigenvals)  # エルミート行列の固有値は実数
            
            result_summary = {
                'success': True,
                'min_eigenvalue': np.min(eigenvals_real),
                'max_eigenvalue': np.max(eigenvals_real),
                'rank': np.sum(eigenvals_real > 1e-8),
                'condition_number': np.max(eigenvals_real) / max(np.min(eigenvals_real), 1e-12),
                'psd_corrected_available': os.path.exists("output/matrix_psd_corrected_real.csv"),
                'is_complex': np.max(np.abs(np.imag(P_result))) > 1e-12
            }
            
            print(f"\n=== 最適化結果要約 ===")
            print(f"成功: {result_summary['success']}")
            print(f"有効ランク: {result_summary['rank']}")
            print(f"PSD修正版利用可能: {result_summary['psd_corrected_available']}")
            
            return result_summary
        else:
            return {'success': False, 'error': 'Output file not found'}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}


# 使用例
"""
実用的な使用方法:

1. 基本的な実行:
   result = run_practical_optimization(basis, target_data, settings)

2. 正則化を強化:
   result = run_practical_optimization(
       basis, target_data, settings, 
       regularization_lambda=1e-5
   )

3. より厳しい目標設定:
   result = run_practical_optimization(
       basis, target_data, settings,
       target_mse=1e-4,
       target_relative_error=0.01
   )

4. 特定成分をゼロに固定:
   # 例：s-d混成を禁止 (4s=0, 3d=1,2,3,4,5なら)
   zero_constraints = [(0,1), (0,2), (0,3), (0,4), (0,5)]
   result = run_practical_optimization(
       basis, target_data, settings,
       zero_constraints=zero_constraints
   )

結果の使用（複素数エルミート行列対応）:
- matrix_real.csv: 最適化結果の実部
- matrix_imag.csv: 最適化結果の虚部（存在する場合のみ）
- matrix.csv: 後方互換性のための実部のみ（従来形式）
- matrix_psd_corrected_real.csv: PSD修正済み実部（利用可能な場合）
- matrix_psd_corrected_imag.csv: PSD修正済み虚部（利用可能な場合）
"""


def generate_orbital_blocking_constraints(
    orbital_types: list[str], 
    blocked_interactions: list[tuple[str, str]]
) -> list[tuple[int, int]]:
    """
    軌道タイプに基づいてゼロ制約を生成
    
    Args:
        orbital_types: 軌道タイプのリスト 例: ['4s', '4px', '4py', '4pz', '3dxy', '3dyz', '3dxz', '3dx2-y2', '3dz2']
        blocked_interactions: 禁止する相互作用のリスト 例: [('4s', '3d'), ('4p', '3d')]
    
    Returns:
        ゼロ制約のリスト [(i,j), ...]
    
    Examples:
        # s-d混成を禁止
        orbital_types = ['4s', '4px', '4py', '4pz', '3dxy', '3dyz', '3dxz', '3dx2-y2', '3dz2']
        blocked = [('4s', '3d')]
        constraints = generate_orbital_blocking_constraints(orbital_types, blocked)
        
        # s-p混成とs-d混成を禁止  
        blocked = [('4s', '4p'), ('4s', '3d')]
        constraints = generate_orbital_blocking_constraints(orbital_types, blocked)
    """
    zero_constraints = []
    
    for orb1_pattern, orb2_pattern in blocked_interactions:
        for i, orb1 in enumerate(orbital_types):
            for j, orb2 in enumerate(orbital_types):
                # パターンマッチング
                if (orb1_pattern in orb1 and orb2_pattern in orb2) or \
                   (orb1_pattern in orb2 and orb2_pattern in orb1):
                    if i != j:  # 対角成分は除外（自己相互作用は許可）
                        zero_constraints.append((i, j))
    
    # 重複を除去
    zero_constraints = list(set(zero_constraints))
    
    print(f"生成されたゼロ制約: {len(zero_constraints)}個")
    for i, j in zero_constraints:
        print(f"  P[{i},{j}] = 0  ({orbital_types[i]} - {orbital_types[j]})")
    
    return zero_constraints


def generate_symmetry_based_constraints(
    orbital_quantum_numbers: list[tuple[int, int, int]],
    forbidden_transitions: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = None
) -> list[tuple[int, int]]:
    """
    量子数に基づく対称性制約を生成
    
    Args:
        orbital_quantum_numbers: [(n,l,m), ...] の量子数リスト
        forbidden_transitions: 禁止遷移のリスト [((n1,l1,m1), (n2,l2,m2)), ...]
    
    Returns:
        ゼロ制約のリスト
        
    Examples:
        # Fe の 4s, 4p, 3d軌道
        quantum_numbers = [
            (4,0,0),     # 4s
            (4,1,-1), (4,1,0), (4,1,1),  # 4p
            (3,2,-2), (3,2,-1), (3,2,0), (3,2,1), (3,2,2)  # 3d
        ]
        
        # s-d直接混成を禁止 (選択則: Δl = ±1)
        forbidden = [
            ((4,0,0), (3,2,-2)), ((4,0,0), (3,2,-1)), 
            ((4,0,0), (3,2,0)), ((4,0,0), (3,2,1)), ((4,0,0), (3,2,2))
        ]
        constraints = generate_symmetry_based_constraints(quantum_numbers, forbidden)
    """
    zero_constraints = []
    
    if forbidden_transitions is not None:
        for qn1, qn2 in forbidden_transitions:
            try:
                i = orbital_quantum_numbers.index(qn1)
                j = orbital_quantum_numbers.index(qn2)
                zero_constraints.append((i, j))
                if i != j:
                    zero_constraints.append((j, i))  # 対称性
            except ValueError:
                print(f"警告: 量子数 {qn1} または {qn2} が見つかりません")
    
    # 自動的な選択則適用（例：Δl ≠ ±1の遷移を禁止）
    # この部分は必要に応じてカスタマイズ
    
    print(f"対称性ベースのゼロ制約: {len(zero_constraints)}個")
    return list(set(zero_constraints))


# 具体的な使用例
"""
# Fe系での典型的な使用例:

# 1. 軌道名ベースでの制約
orbital_names = ['4s', '4px', '4py', '4pz', '3dxy', '3dyz', '3dxz', '3dx2-y2', '3dz2']

# s-d混成のみ禁止
zero_constraints = generate_orbital_blocking_constraints(
    orbital_names, [('4s', '3d')]
)

# 2. 量子数ベースでの制約  
quantum_numbers = [
    (4,0,0),     # 4s
    (4,1,-1), (4,1,0), (4,1,1),  # 4p
    (3,2,-2), (3,2,-1), (3,2,0), (3,2,1), (3,2,2)  # 3d
]

forbidden_transitions = [
    ((4,0,0), (3,2,m)) for m in [-2,-1,0,1,2]  # s-d直接遷移を禁止
]

zero_constraints = generate_symmetry_based_constraints(
    quantum_numbers, forbidden_transitions
)

# 3. 最適化の実行
result = run_practical_optimization(
    basis, target_data, settings,
    zero_constraints=zero_constraints,
    regularization_lambda=1e-6
)
"""
    
    