import json
import os

# 路径到你的JSON文件的文件夹
directory_path = '/data1/haihongzhao/DSAA6000I-Final-Project-Group-7/data_for_LLM/RAWFC/raw_data/test'

# 输出文件夹
output_folder = '/data1/haihongzhao/DSAA6000I-Final-Project-Group-7/data_for_LLM/RAWFC'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
output_file = os.path.join(output_folder, 'test_use.jsonl')


# 映射函数
def map_label(label):
    true_labels = ['True', 'true']  # 可以识别为True的标签列表
    if label in true_labels:
        return 'True'
    else:
        return 'False'


# 处理JSON文件并提取需要的数据
def process_json_files(directory):
    data_list = []  # 用于存储处理后的数据集

    # 遍历文件夹中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)

            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                # 提取必要的字段
                claim = data.get("claim", "")
                label = map_label(data.get("label", ""))
                explain = data.get("explain", "").replace('"', '\\"')  # 确保引号被正确地转义

                # 构造evidence、claim和response字段
                response = f"According to our knowledge and the given information, we think that the claim is {label}."

                # 将数据添加到列表
                data_list.append({"evidence": explain, "claim": claim, "response": response})

    return data_list


# 处理文件夹内的所有JSON文件
dataset = process_json_files(directory_path)

# 将数据集保存到指定的JSON文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)
# 输出提示
print(f'All data has been processed and saved to {output_file}.')