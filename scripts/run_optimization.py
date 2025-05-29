import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.flows.optimization_flow import optimization_pipeline

def main():
    """最適化パイプラインの実行エントリポイント"""
    parser = argparse.ArgumentParser(description='非線形最適化パイプライン')
    parser.add_argument('--data', default='data/input/data.xplor', help='入力データのパス')
    parser.add_argument('--config', default='data/input/settings.yaml', help='設定ファイルのパス')
    parser.add_argument('--output-dir', default='data/output', help='結果の出力ディレクトリ')
    parser.add_argument('--run-name', default=None, help='実行の名前（デフォルトは日時）')
    
    args = parser.parse_args()
    
    # パイプラインの実行
    optimization_pipeline(
        data_path=args.data,
        config_path=args.config,
        output_dir=args.output_dir,
        run_name=args.run_name
    )
    
    # print("\n最終結果のサマリー:")
    # print(f"実行名: {result['run_name']}")
    # print(f"結果ディレクトリ: {result['run_dir']}")
    # print(f"最適化されたパラメータ: {result['optimized_parameters']}")
    # print(f"最終誤差: {result['final_error']}")
    # # print(f"詳細結果: {result['result_path']}")

if __name__ == "__main__":
    main()