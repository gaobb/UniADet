
import os
import csv
import re
from collections import defaultdict
from tabulate import tabulate

def general_parse_log(log_file):
    """解析单个日志文件，只获取最后一次运行的表格数据"""
    with open(log_file, 'r', encoding='utf-8') as file:
        content = file.read()
    
    '''
    # 匹配表格的正则表达式
    table_pattern = re.compile(
        r'\|\s*Name\s*\|\s*I-AUROC\s*\|\s*I-AP\s*\|\s*I-F1max\s*\|\s*P-AUROC\s*\|\s*P-AP\s*\|\s*P-F1max\s*\|\s*P-AUPRO\s*\|'
        r'(?:.*?\n)*?'
        r'\|\s*[-:\s]+\s*\|\s*[-:\s]+\s*\|\s*[-:\s]+\s*\|\s*[-:\s]+\s*\|\s*[-:\s]+\s*\|\s*[-:\s]+\s*\|\s*[-:\s]+\s*\|\s*[-:\s]+\s*\|'
        r'(.*?)(?=\n\n|\n\s*\n|\Z)',
        re.DOTALL
    )
    
    # 匹配数据行的正则表达式
    data_row_pattern = re.compile(
        r'\|\s*(\w+)\s*\|\s*([\d.]+|nan)\s*\|\s*([\d.]+|nan)\s*\|\s*([\d.]+|nan)\s*\|\s*([\d.]+|nan)\s*\|\s*([\d.]+|nan)\s*\|\s*([\d.]+|nan)\s*\|\s*([\d.]+|nan)\s*\|'
    )
    
    # 查找所有表格
    all_tables = table_pattern.findall(content)
    
    if not all_tables:
        return None
    
    # 只获取最后一个表格的数据
    last_table = all_tables[-1]
    '''
    # -----------------------------------------------------------------
    # ① 表头正则（和您原来的一模一样，只是单独抽出来）
    # -----------------------------------------------------------------
    header_pattern = re.compile(
        r'\|+\s*Name\s*\|\s*I-AUROC\s*\|\s*I-AP\s*\|\s*I-F1max\s*\|'
        r'\s*P-AUROC\s*\|\s*P-AP\s*\|\s*P-F1max\s*\|\s*P-AUPRO\s*\|\s*\n'
        r'\|:?[-:]+\|:?[-:]+\|:?[-:]+\|:?[-:]+\|:?[-:]+\|:?[-:]+\|:?[-:]+\|:?[-:]+\|\s*\n',
        re.IGNORECASE,
    )

    # -----------------------------------------------------------------
    # ② 行正则（和您原来的一模一样）
    # -----------------------------------------------------------------
    '''
    row_pattern = re.compile(
        r'\|+\s*(\w+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|'
        r'\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|',
        re.IGNORECASE,
    )
    '''

    # 2. 行模式：允许非数字值（如 nan，-）和灵活的空格
    row_pattern = re.compile(
        r'\|+\s*(\S+)\s*\|'  # \S+ 匹配非空白字符（更灵活的名称）
        r'\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|NaN|N/A|-)\s*\|'  # 数值或特殊标记
        r'\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|NaN|N/A|-)\s*\|'
        r'\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|NaN|N/A|-)\s*\|'
        r'\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|NaN|N/A|-)\s*\|'
        r'\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|NaN|N/A|-)\s*\|'
        r'\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|NaN|N/A|-)\s*\|'
        r'\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|NaN|N/A|-)\s*\|',
        re.IGNORECASE,
    )

    # -----------------------------------------------------------------
    # ③ 找到 **最后一次** 表头出现的位置
    # -----------------------------------------------------------------
    header_matches = list(header_pattern.finditer(content))
    if not header_matches:
        # 没有表格，直接返回空列表
        return []

    # 取最后一次出现的表头结束位置（即表格内容的起始位置）
    last_header_end = header_matches[-1].end()

    # -----------------------------------------------------------------
    # ④ 在 “最后一次表头之后的子串” 中匹配所有行
    # -----------------------------------------------------------------
    table_section = content[last_header_end:]   # 只保留最后一次表格的正文
    performance_data = []

    for m in row_pattern.finditer(table_section):
        class_name = m.group(1)
        metrics = [float(m.group(i)) for i in range(2, 9)]
        # 过滤掉平均值行（Avg, Mean等）
        if class_name.lower() in ['avg', 'mean', 'average']:
            class_name = 'Mean'
        performance_data.append((class_name, metrics))
        

    '''
    # 解析最后一个表格中的数据行
    for match in data_row_pattern.finditer(last_table):
        class_name = match.group(1)
        
        # 处理nan值，将其设为None或0
        metrics = []
        for i in range(2, 9):
            value = match.group(i)
            if value == 'nan':
                metrics.append(None)  # 或者使用 0.0
            else:
                try:
                    metrics.append(float(value))
                except ValueError:
                    metrics.append(None)
        
        # 过滤掉平均值行（Avg, Mean等）
        if class_name.lower() in ['avg', 'mean', 'average']:
            class_name = 'Mean'
        performance_data.append((class_name, metrics))
    '''
    return performance_data



def real_iad_parse_log(log_file):
    with open(log_file, 'r') as file:
        log_content = file.read()

    # 更精确的正则表达式来匹配表格数据
    pattern = re.compile(
        r'\|+\s*Name\s*\|\s*I-AUROC\s*\|\s*I-AP\s*\|\s*I-F1max\s*\|\s*P-AUROC\s*\|\s*P-AP\s*\|\s*P-F1max\s*\|\s*P-AUPRO\s*\|\s*\n'
        r'\|:?[-:]+\|:?[-:]+\|:?[-:]+\|:?[-:]+\|:?[-:]+\|:?[-:]+\|:?[-:]+\|:?[-:]+\|\s*\n'
        r'\|+\s*(\w+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|'
    )

    performance_data = []

    for match in pattern.finditer(log_content):
        class_name = match.group(1)
        metrics = [float(match.group(i)) for i in range(2, 9)]
        performance_data.append((class_name, metrics))

    return performance_data

def calculate_averages(performance_data):
    if not performance_data:
        return None

    # 初始化累加器
    sums = [0.0] * 7
    count = 0

    for _, metrics in performance_data:
        for i in range(7):
            sums[i] += metrics[i]
        count += 1

    # 计算平均值
    averages = [sums[i] / count for i in range(7)]
    return averages

def format_output(performance_data):
    headers = ['Name', 'I-AUROC', 'I-AP', 'I-F1max', 'P-AUROC', 'P-AP', 'P-F1max', 'P-AUPRO']
    table = []

    # 添加每个类别的数据
    class_names = []
    for class_name, metrics in performance_data:
        table.append([class_name] + metrics)
        class_names.append(class_name)

    # 计算并添加平均值
    if 'Mean' not in class_names:
        averages = calculate_averages(performance_data)
        if averages:
            table.append(['Mean'] + averages)

    return tabulate(table, headers=headers, floatfmt=".1f") #tablefmt="grid"


def write_performance_to_csv(performance_data, csv_file_path):
    """
    将解析的performance_data写入CSV文件
    
    Args:
        performance_data: 解析得到的性能数据，格式为 [(class_name, [metrics]), ...]
        csv_file_path: CSV文件保存路径
    """
    headers = ['Name', 'I-AUROC', 'I-AP', 'I-F1max', 'P-AUROC', 'P-AP', 'P-F1max', 'P-AUPRO']
    class_names = []
    try:
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # 写入表头
            writer.writerow(headers)
            
            # 写入每个类别的数据
            for class_name, metrics in performance_data:
                row = [class_name] + metrics
                writer.writerow(row)
                class_names.append(class_name)
            
            # 写入平均值
            if 'Mean' not in class_names:
                averages = calculate_averages(performance_data)
                if averages:
                    #avg_row = ['Mean'] + averages
                    #writer.writerow(avg_row)
                    mean_row = ["Mean"] + [f"{v:.1f}" for v in averages]
                    writer.writerow(mean_row)
            
                
        print(f"性能数据已成功写入CSV文件: {csv_file_path}")
        return True
        
    except Exception as e:
        print(f"写入CSV文件时发生错误: {str(e)}")
        return False


if __name__ == "__main__":
    save_path = './results/'
    
    train_dataset = 'visa'
    test_dataset = 'RealIAD'

    #pretrained_model = "ViT-L/14@336px"
    #imge_size = 518
    #pretrained_model = "ViT-L-16-DINOv3"
    # imge_size = 512

    pretrained_model = "ViT-L-14-DINOv2Reg"
    imge_size = 518
    
    for test_dataset in ['RealIAD', 'medical-cls', 'medical-seg', 'mvtec', 'MPDD', 'DTD-Synthetic', 'SDD', 'BTAD', 'Real-IAD-Variety', "MVTec3D"]:
        for shot in [0, 1, 2, 4]:
            if shot == 0:
                seeds = [10]
            else:   
                seeds = [10, 20, 30]
            for seed in seeds:
                log_file = save_path + 'train_on_visa_zs_${pretrained_model}_imagesize${imge_size}_1015/${test_dataset}_${seed}seed_test_${shot}shot_log_2025-11-05.txt'
                print(log_file)
                if 'RealIAD' in log_file:
                    performance_data = real_iad_parse_log(log_file)
                else:
                    performance_data = general_parse_log(log_file)

                formatted_table = format_output(performance_data)
                print(formatted_table)

                # 将结果写入CSV文件
                log_filename = os.path.basename(log_file)
                exp_set_data = log_filename.split('_log_')[0].split('/')[-1]
                exp_set_data = exp_set_data.replace('_log.txt', '')
                exp_set_model = os.path.dirname(log_file).split('/')[-1]

                csv_path = os.path.join(save_path, exp_set_model)
                print(csv_path)
                if os.path.exists(csv_path) is False:
                    os.makedirs(csv_path)
                csv_file = os.path.join(csv_path, f'{exp_set_data}.csv')
                
                write_performance_to_csv(performance_data, csv_file)
                