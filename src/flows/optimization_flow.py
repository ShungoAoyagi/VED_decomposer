import os
from prefect import flow
import yaml
import numpy as np

from src.tasks.pre_processing import create_orbitals, import_settings, load_data, calculate_Zeff
from src.tasks.after_processing import analyze_gamma
from src.tasks.data_processing import fit_gamma_sdp
from src.models import create_objective_function

@flow(name="非線形最適化パイプライン")
def optimization_pipeline(
    data_path,
    config_path=None,
    output_dir="data/output",
    run_name=None
):
    # 設定の読み込み
    if config_path is None:
        config_path = "config/config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 実行名の設定
    if run_name is None:
        from datetime import datetime
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # パラメータの取得
    initial_params = np.array(config.get("initial_params", [0.1, 0.1, 0.1]))
    # optimization_method = config.get("optimization_method", "BFGS")
    optimization_options = config.get("optimization_options", {})
    
    # データ読み込み
    settings = import_settings(config_path)
    data = load_data(data_path)

    # 孤立原子での軌道の計算
    Zeff = calculate_Zeff(data)
    
    # データ前処理
    processed_data = create_orbitals(data)
    
    # 中間結果の保存（オプション）
    # processed_data_path = os.path.join(run_dir, "processed_data.csv")
    # processed_data.to_csv(processed_data_path, index=False)
    
    # 目的関数の作成
    objective_fn = create_objective_function(
        processed_data
    )
    
    # 最適化の実行
    result = fit_gamma_sdp(
        objective_fn=objective_fn,
        initial_params=initial_params,
        options=optimization_options
    )
    
    # 結果の保存
    # output_path = os.path.join(run_dir, "optimization_results.json")
    # result_path = analyze_gamma(result, output_path)
    
    # 結果の可視化（オプション）
    # if config.get("create_visualizations", True):
    #     plot_path = os.path.join(run_dir, "optimization_plot.png")
    #     visualize_results(result, processed_data, plot_path)
    
    return {
        "run_name": run_name,
        "run_dir": data_path,
        "optimized_parameters": result.x.tolist(),
        "final_error": float(result.fun),
        # "result_path": result_path
    }