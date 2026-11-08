import os
import csv
import re
from tabulate import tabulate


def parse_markdown_table(content: str) -> tuple[list[str], list[list]]:
    """
    通用 Markdown 表格解析器
    支持两种格式：
    1. 单个大表格（一个表头，多行数据）
    2. 多个小表格（每个类别独立表头，每表格一行数据）

    注意：遇到重复类别名称时，保留最后一次结果（覆盖）

    Returns:
        (headers, rows): 表头列表和数据行列表
    """
    lines = content.strip().split('\n')

    all_headers = None
    row_dict = {}  # 使用字典存储，key为类别名称，重复时自动覆盖
    row_order = []  # 保持类别出现顺序

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # 检测表头行：以 | 开头且包含 Name
        if line.startswith('|') and 'Name' in line:
            # 检查下一行是否为分隔符行
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if re.match(r'^\|[\s:|-]+\|$', next_line):
                    # 解析表头
                    headers = [h.strip() for h in line.split('|') if h.strip()]

                    # 保存第一次遇到的表头
                    if all_headers is None:
                        all_headers = headers

                    # 解析后续数据行
                    j = i + 2
                    while j < len(lines):
                        data_line = lines[j].strip()
                        # 数据行以 | 开头，且不是分隔符行
                        if data_line.startswith('|') and not re.match(r'^\|[\s:|-]+\|$', data_line):
                            cells = [c.strip() for c in data_line.split('|') if c.strip()]
                            if len(cells) >= len(headers):
                                cells = cells[:len(headers)]  # 截取匹配的列数
                            if cells and cells[0].lower() not in ['name']:
                                name = cells[0]
                                # 重复类别：覆盖旧值，保留最后一次结果
                                if name not in row_dict:
                                    row_order.append(name)
                                row_dict[name] = cells
                            j += 1
                        else:
                            break
                    i = j
                    continue
        i += 1

    # 按出现顺序返回结果
    all_rows = [row_dict[name] for name in row_order]
    return all_headers or [], all_rows


def parse_log_file(log_file: str) -> tuple[list[str], list[list]]:
    """
    解析日志文件中的表格数据

    Args:
        log_file: 日志文件路径

    Returns:
        (headers, rows): 表头列表和数据行列表（数值已转换）
    """
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    headers, raw_rows = parse_markdown_table(content)

    if not headers or not raw_rows:
        return [], []

    # 转换数据类型：第一列为名称，其余为数值
    rows = []
    for raw_row in raw_rows:
        row = [raw_row[0]]  # 名称列
        for val in raw_row[1:]:
            try:
                row.append(float(val))
            except ValueError:
                row.append(float('nan'))  # nan, N/A, - 等
        rows.append(row)

    return headers, rows


def calculate_mean_row(headers: list[str], rows: list[list]) -> list:
    """计算平均值行"""
    if not rows:
        return []

    # 检查是否已有 Mean 行
    for row in rows:
        if str(row[0]).lower() in ['mean', 'avg', 'average']:
            return []

    num_cols = len(headers)
    mean_row = ['Mean']

    for col_idx in range(1, num_cols):
        values = []
        for row in rows:
            if col_idx < len(row) and not (isinstance(row[col_idx], float) and row[col_idx] != row[col_idx]):
                values.append(row[col_idx])
        if values:
            mean_row.append(sum(values) / len(values))
        else:
            mean_row.append(float('nan'))

    return mean_row


def format_output(headers: list[str], rows: list[list]) -> str:
    """格式化输出表格"""
    table_data = []
    for row in rows:
        table_data.append(row)

    # 添加平均值行
    mean_row = calculate_mean_row(headers, rows)
    if mean_row:
        table_data.append(mean_row)

    return tabulate(table_data, headers=headers, floatfmt=".1f")


def write_to_csv(headers: list[str], rows: list[list], csv_path: str) -> bool:
    """写入 CSV 文件"""
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for row in rows:
                formatted = [row[0]] + [f"{v:.1f}" if isinstance(v, float) else v for v in row[1:]]
                writer.writerow(formatted)

            # 添加平均值行
            mean_row = calculate_mean_row(headers, rows)
            if mean_row:
                formatted = [mean_row[0]] + [f"{v:.1f}" if isinstance(v, float) else v for v in mean_row[1:]]
                writer.writerow(formatted)

        return True
    except Exception as e:
        print(f"写入 CSV 失败: {e}")
        return False


def process_log(log_file: str, output_csv: str = None) -> None:
    """
    处理单个日志文件

    Args:
        log_file: 日志文件路径
        output_csv: 输出 CSV 路径（可选，默认同名 .csv）
    """
    if not os.path.exists(log_file):
        print(f"文件不存在: {log_file}")
        return

    headers, rows = parse_log_file(log_file)

    if not headers:
        print(f"未找到有效表格: {log_file}")
        return

    # 输出路径
    if output_csv is None:
        output_csv = os.path.splitext(log_file)[0] + '.csv'

    # 打印表格
    print(f"\n解析完成: {len(rows)} 个类别, {len(headers)} 个指标")
    print(f"输出文件: {output_csv}\n")
    print(format_output(headers, rows))

    # 写入 CSV
    write_to_csv(headers, rows, output_csv)
    print(f"\n已保存至: {output_csv}")


# 兼容旧接口
def general_parse_log(log_file):
    """兼容旧版接口"""
    headers, rows = parse_log_file(log_file)
    return [(row[0], row[1:]) for row in rows]


def real_iad_parse_log(log_file):
    """兼容旧版接口"""
    return general_parse_log(log_file)


def calculate_averages(performance_data):
    """兼容旧版接口"""
    if not performance_data:
        return None
    num_metrics = len(performance_data[0][1])
    sums = [0.0] * num_metrics
    count = 0
    for _, metrics in performance_data:
        for i, v in enumerate(metrics):
            if not (isinstance(v, float) and v != v):  # 排除 nan
                sums[i] += v
        count += 1
    return [s / count for s in sums] if count > 0 else None


def format_output_legacy(performance_data):
    """兼容旧版接口"""
    if not performance_data:
        return ""
    num_metrics = len(performance_data[0][1])
    headers = ['Name'] + [f'Metric_{i}' for i in range(num_metrics)]
    rows = [[name] + list(metrics) for name, metrics in performance_data]
    mean_row = calculate_averages(performance_data)
    if mean_row:
        rows.append(['Mean'] + mean_row)
    return tabulate(rows, headers=headers, floatfmt=".1f")


def write_performance_to_csv(performance_data, csv_file_path):
    """兼容旧版接口"""
    if not performance_data:
        return False
    num_metrics = len(performance_data[0][1])
    headers = ['Name'] + [f'Metric_{i}' for i in range(num_metrics)]
    rows = [[name] + list(metrics) for name, metrics in performance_data]
    return write_to_csv(headers, rows, csv_file_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='解析测试日志并生成 CSV')
    args = parser.parse_args()

    # 批量处理示例
    train_dataset = 'visa'
    

    #pretrained_model = "ViT-L/14@336px"
    #imge_size = 518
    #pretrained_model = "ViT-L-16-DINOv3"
    # imge_size = 512

    pretrained_model = "ViT-L-14-DINOv2Reg"
    imge_size = 518


    save_path = './results/'

    for test_dataset in ['RealIAD', 'Real-IAD-Variety', 'medical', 'medical-cls', 'visa']:
        for shot in [0, 1, 2, 4]: #, 1, 2, 4]:
            seeds = [10] if shot == 0 else [10, 20, 30]
            for seed in seeds:
                log_file = f'{log_path}train_on_{train_dataset}_zs_{pretrained_model}_imagesize{image_size}_1015/{test_dataset}_{seed}seed_test_{shot}shot_log_2025.txt' #_2025-11-04.txt'
                print(f"\n处理: {log_file}")
                if os.path.exists(log_file):
                    print(f"\n处理: {log_file}")
                    headers, rows = parse_log_file(log_file)
                    if headers:
                        print(format_output(headers, rows))
                        # 保存 CSV
                        csv_dir = os.path.join(save_path, os.path.dirname(log_file).split('/')[-1])
                        os.makedirs(csv_dir, exist_ok=True)
                        csv_file = os.path.join(csv_dir, f'{test_dataset}_{seed}seed_{shot}shot.csv')
                        write_to_csv(headers, rows, csv_file)