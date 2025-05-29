import os
from prefect import flow
import yaml
import numpy as np

from src.tasks.pre_processing import import_settings, load_data
from src.tasks.data_processing.main_loop import main_loop

@flow(name="非線形最適化パイプライン")
def optimization_pipeline(
    data_path,
    config_path=None,
    output_dir="data/output",
    run_name=None
):
    # 設定の読み込み
    if config_path is None:
        config_path = "config/settings.yaml"
    
    # 実行名の設定
    if run_name is None:
        from datetime import datetime
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    # load settings
    settings = import_settings(config_path)
    
    # load data
    data = load_data(data_path, settings)

    main_loop(data, settings)
    
    
    # 結果の保存
    # output_path = os.path.join(run_dir, "optimization_results.json")
    # result_path = analyze_gamma(result, output_path)
    
    # 結果の可視化（オプション）
    # if config.get("create_visualizations", True):
    #     plot_path = os.path.join(run_dir, "optimization_plot.png")
    #     visualize_results(result, processed_data, plot_path)
    
    # return {
    #     "run_name": run_name,
    #     "run_dir": data_path,
    #     "optimized_parameters": result.x.tolist(),
    #     "final_error": float(result.fun),
    #     # "result_path": result_path
    # }