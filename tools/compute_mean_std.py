import pandas as pd
import os
from typing import List, Optional, Tuple

# ---- 函数 1: 提取单个CSV文件信息及表头 ----
def extract_mean_from_csv(file_path: str, dataset_name: str) -> Optional[Tuple[pd.DataFrame, List[str]]]:
    """
    从单个CSV文件中读取数据，找到'Mean'行，并返回该行数据及文件的原始表头。

    Args:
        file_path (str): CSV文件的完整路径。
        dataset_name (str): 用于替换'Mean'的数据集名称。

    Returns:
        Optional[Tuple[pd.DataFrame, List[str]]]: 
        如果成功，返回一个元组 (包含单行数据的DataFrame, 原始表头列表)；否则返回 None。
    """
    if not os.path.exists(file_path):
        print(f"警告: 文件 '{file_path}' 不存在，已跳过。")
        return None

    try:
        # 尝试读取文件
        df = pd.read_csv(file_path)
        header = df.columns.tolist()  # 提取原始表头
        
        # 查找 'Name' 列中值为 'Mean' 的行
        mean_row = df[df['Name'] == 'Mean'].copy()

        if mean_row.empty:
            print(f"警告: 在文件 '{file_path}' 中未找到 'Mean' 行。")
            return None
        
        # 替换数据集名称
        mean_row['Name'] = dataset_name
        print(f"成功提取 '{dataset_name}' 的Mean数据。")
        
        # 返回数据和表头
        return mean_row, header

    except Exception as e:
        print(f"处理文件 '{file_path}' 时发生错误: {e}")
        return None


# ---- 函数 2: 结果整合函数 (带表头检查、重排序和平均值计算) ----
def integrate_results(results_list: List[pd.DataFrame], reference_header: List[str]) -> Optional[pd.DataFrame]:
    """
    将多个单行DataFrame合并，检查并统一列序，并计算所有数据集的平均值作为最后一行。
    自动处理数值列中的 '平均值 ± 标准差' 字符串格式，并计算平均值之间的标准差。

    Args:
        results_list (List[pd.DataFrame]): 从各个文件中提取出的结果列表。
        reference_header (List[str]): 用于统一列顺序的参考表头。

    Returns:
        Optional[pd.DataFrame]: 返回合并并包含平均值行的DataFrame；如果列表为空，则返回None。
    """
    if not results_list:
        print("未能从任何文件中提取到有效数据。")
        return None
        
    reordered_results = []
    for df_item in results_list:
        if set(df_item.columns) != set(reference_header):
            dataset_name = df_item['Name'].iloc[0]
            print(f"!!严重警告!!: 数据集 '{dataset_name}' 的表头与参考表头不匹配，将跳过此条目。")
            print(f"  参考表头: {reference_header}")
            print(f"  当前表头: {df_item.columns.tolist()}")
            continue
        
        reordered_df = df_item[reference_header]
        reordered_results.append(reordered_df)

    if not reordered_results:
        print("经过表头检查后，没有剩余的有效数据可供整合。")
        return None

    # 将所有列顺序统一后的DataFrame进行合并 (包含 'XX.X ± XX.X' 字符串)
    summary_df = pd.concat(reordered_results, ignore_index=True)

    # --- **修改后的代码**: 处理字符串格式并计算平均值和平均值的标准差 ---
    print("开始处理字符串格式数据并计算所有数据集的平均值和平均值之间的标准差...")
    
    # 1. 识别需要清洗的列（排除 'Name'）
    cols_to_clean = [col for col in summary_df.columns if col != 'Name']
    
    # 初始化用于计算 Mean 的 DataFrame
    mean_calc_df = summary_df.copy()
    
    # 2. 清洗数据：提取平均值部分并转换为浮点数
    for col in cols_to_clean:
        try:
            split_data = mean_calc_df[col].astype(str).str.split(' ± ', expand=True)
            
            # 提取 Mean (左侧部分)
            # errors='coerce' 会将无法转换的值设为 NaN
            mean_calc_df[col] = pd.to_numeric(split_data[0], errors='coerce')
        except Exception:
             # 如果不是预期格式，则无法计算，设为 NaN
             mean_calc_df[col] = pd.NA
             
    # 3. 计算所有数据集 Mean 的平均值 (Overall Mean)
    overall_mean_series = mean_calc_df.mean(numeric_only=True)
    
    # 4. **关键改动**: 计算所有数据集 Mean 的标准差 (Standard Deviation of the Means)
    # 这衡量了结果在不同数据集之间的离散程度
    overall_stddev_series = mean_calc_df.std(numeric_only=True)
    
    if not overall_mean_series.empty:
        # 5. 格式化总平均值行
        
        # Mean 四舍五入到一位小数
        overall_mean_rounded = overall_mean_series.round(1)
        # StdDev 四舍五入到一位小数
        overall_stddev_rounded = overall_stddev_series.round(1)
        
        # 格式化回 'XX.X ± Y.Y' 的字符串格式
        overall_mean_formatted = {}
        for col in cols_to_clean:
            mean_value = overall_mean_rounded.get(col)
            stddev_value = overall_stddev_rounded.get(col)
            
            # 检查是否有有效的平均值
            if pd.notna(mean_value):
                # 检查是否有有效的标准差，如果没有则默认使用 0.0
                stddev_to_use = stddev_value if pd.notna(stddev_value) else 0.0
                
                # 组合平均值和其标准差
                #overall_mean_formatted[col] = f"{mean_value:.1f} ± {stddev_to_use:.1f}"
                overall_mean_formatted[col] = f"{mean_value:.1f}"
            else:
                overall_mean_formatted[col] = '' # 无法计算的列留空
            
        overall_mean_formatted['Name'] = 'Mean'
        
        # 6. 创建单行 DataFrame 并追加到汇总表
        overall_mean_df = pd.DataFrame([overall_mean_formatted], columns=reference_header)
        summary_df = pd.concat([summary_df, overall_mean_df], ignore_index=True)
        
        print("已将所有数据集的平均值 ('Overall Mean') 添加到结果的最后一行，并计算了平均值的标准差。")
    else:
        print("警告: 未找到任何数值列可供计算平均值。")
    # ----------------------------------------------------------------------
    
    return summary_df


# ---- 函数 3: CSV写入函数 (无变化) ----
def write_summary_to_csv(summary_df: Optional[pd.DataFrame], output_file: str):
    """
    将汇总后的DataFrame写入到CSV文件中。
    """
    if summary_df is None or summary_df.empty:
        print("没有可写入的数据，已跳过文件生成。")
        return

    summary_df.to_csv(output_file, index=False)
    print("-" * 30)
    print(f"任务完成！汇总结果已保存至 '{output_file}'。")
    print("生成内容预览:")
    # 为了更好地在终端展示，这里只打印前几行
    print(summary_df.head(10)) 


# ---- 主执行逻辑 (main) ----
if __name__ == '__main__':
    # --- 配置区域 ---
    # 这里的配置和路径设置依赖于您的文件系统结构，这里保留原始逻辑
    train_dataset='visa'
    pre_model='ViT-L-14-DINOv2Reg'
    pre_model='ViT-L-16-DINOv3'
    pre_model='ViT-L-14@336px'
    
    shot = 0
    seed = 10
    if pre_model in ['ViT-L-16-DINOv3']:
        imagesize=512
    else:
        imagesize=518

    datasets = ['mvtec', 'visa', 'BTAD', 'MVTec3D', 'DTD-Synthetic', 'SDD', 'MPDD', 'RealIAD']
    output_csv = f'./Results/{pre_model}_{shot}shot_industry.csv'
    
    print("开始处理流程...")
    
    all_extracted_means = []
    reference_header = None # 用于存储第一个成功读取的文件的表头

    # 1. 循环调用提取函数，收集所有结果并确立参考表头
    for name in datasets:
        # 复杂的文件路径逻辑，根据您的原始代码保留
        if name == 'visa':
            train_dataset='mvtec'
        else:
            train_dataset='visa'

        results_path = f'./Results/train_on_{train_dataset}_zs_{pre_model}_imagesize{imagesize}_1015'
    
        
        if shot == 0:
            data_csv_path = os.path.join(results_path, f"{name}_{seed}seed_test_{shot}shot.csv")
        else:
            data_csv_path = os.path.join(results_path, f"{name}_{shot}shot_mean_std.csv")

        result = extract_mean_from_csv(file_path=data_csv_path, dataset_name=name)
        
        if result:
            mean_row, current_header = result
            
            # 如果这是第一个成功处理的文件，将其表头设为参考标准
            if reference_header is None:
                reference_header = current_header
                print(f"-> 已确立参考表头为: {reference_header}")
            
            all_extracted_means.append(mean_row)
    
    print("\n所有文件提取完毕，开始整合结果...")
    
    # 2. 调用整合函数（传入数据列表和参考表头）
    # 现在这个函数将自动处理字符串，计算平均值和平均值间的标准差，并四舍五入到一位小数后添加为最后一行
    final_summary = integrate_results(all_extracted_means, reference_header)
    
    # 3. 调用写入函数
    write_summary_to_csv(final_summary, output_csv)
    
    print("\n流程结束。")