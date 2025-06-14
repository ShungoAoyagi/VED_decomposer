import os
import numpy as np
from src.tasks.pre_processing.settings import Settings

def custom_scientific_notation(value: float | complex, max_digits: int = 6) -> str:
    """
    Convert a floating point number to scientific notation with format 0.XXXE+YY
    
    Args:
        value: Input floating point number
        
    Returns:
        String representation in format 0.XXXE+YY
    """
    if value == 0:
        return f"{0.0:12.4E}"
    
    if isinstance(value, complex):
        # 虚数部の符号を適切に処理
        if np.abs(value.imag) < 1e-5:
            return f"{value.real:.{max_digits}f}"
        elif value.imag >= 0:
            return f"{value.real:.{max_digits}f}+{value.imag:.{max_digits}f}j"
        else:
            return f"{value.real:.{max_digits}f}{value.imag:.{max_digits}f}j"
    
    # 絶対値を取る
    abs_value = abs(value)
    
    # 元の値の符号を保持
    sign = -1 if value < 0 else 1
    
    # 指数部分を計算
    exponent = int(np.floor(np.log10(abs_value))) + 1
    
    # 仮数部分を計算（0.1 <= abs_mantissa < 1.0 の形式に）
    mantissa = abs_value / (10 ** exponent)
    
    
    # 結果のフォーマット（0.XXXX形式になるように整形）
    # 符号付き仮数部と指数部を手動でフォーマット
    mantissa_str = f"{mantissa:.{max_digits}f}"  # 小数点以下4桁
    exp_sign = "+" if exponent >= 0 else "-"
    exp_abs = abs(exponent)
    
    # 最終的な形式に整形
    result = f"{"-" if sign == -1 else ""}{mantissa_str}E{exp_sign}{exp_abs:02d}"
    
    # 全体の幅を調整（必要に応じてスペースで埋める）
    result = f"{result:>{max_digits + 7}}"
    
    return result

def make_xplor(data: np.ndarray, output_path: str, output_name: str, settings: Settings) -> None:
    """
    make xplor file
    """
    if os.path.exists(output_path):
        os.remove(output_path)
    
    length = data.shape

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("\n")
        f.write("    1\n")
        f.write(f"{output_name}\n")
        f.write(f"  {length[0]}       0       {length[0] - 1}       {length[1]}       0       {length[1] - 1}       {length[2]}       0       {length[2] - 1}\n")
        f.write(f" {custom_scientific_notation(settings.lattice_params[0], 4)} {custom_scientific_notation(settings.lattice_params[1], 4)} {custom_scientific_notation(settings.lattice_params[2], 4)} {custom_scientific_notation(settings.lattice_params[3], 4)} {custom_scientific_notation(settings.lattice_params[4], 4)} {custom_scientific_notation(settings.lattice_params[5], 4)}\n")
        f.write(f"ZYX\n ")
        for i in range(length[2]):
            count = 0
            if (i < 10):
                f.write(f"   {i} \n ")
            elif (i < 100):
                f.write(f"  {i} \n ")
            else:
                f.write(f" {i} \n ")
            for j in range(length[1]):
                for k in range(length[0]):
                    f.write(f" {custom_scientific_notation(data[k, j, i])} ")
                    if count % 5 == 4:
                        f.write("\n ")
                    count += 1
            if count % 5 != 0:
                f.write("\n ")
    return