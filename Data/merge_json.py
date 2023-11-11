import json
import os

# 路径到你的JSON文件的文件夹
# directory_path_1 = './Data/LIAR-RAW/train.json'
# directory_path_2 = './Data/RAWFC/train.json'
directory_path_1 = './Data/LIAR-RAW/val.json'
directory_path_2 = './Data/RAWFC/val.json'

# 输出文件夹
output_folder = './Data'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# output_file = os.path.join(output_folder, 'train_ins.json')
output_file = os.path.join(output_folder, 'eval_ins.json')

def merge_data_files(dir_list:list):
    data_list = []
    total_len = 0
    for index, dir in enumerate(dir_list):

        # 读取JSON文件
        with open(dir, 'r', encoding='utf-8') as file:
            data = json.load(file)        
            print(len(data))
            total_len+=len(data)
            data_list+=data
    
    if(total_len == len(data_list)):
        print('所有元素都成功合并')
    
    return data_list

merged_data= merge_data_files([directory_path_1,directory_path_2])

with open(output_file, "a") as json_file:
    for item in merged_data:
        # 将单个字典写入文件，并在字典之间加上逗号和换行符
        json.dump(item, json_file)
        json_file.write("\n")

# 输出提示
print(f'All data has been processed and saved to {output_file}.')
