import json
import os

# 路径到你的JSON文件的文件夹
directory_path_1 = './data_for_LLM/LIAR-RAW/test_use.jsonl'
directory_path_2 = './data_for_LLM/RAWFC/test_use.jsonl'

# 输出文件夹
output_folder = './data_for_LLM/data'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_file = os.path.join(output_folder, 'test_use.jsonl')

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
        json_string = json.dumps(item)
        json_file.write(json_string) #按照字符串写入数据
        # json.dump(item, json_file)
        json_file.write("\n")

# 输出提示
print(f'All data has been processed and saved to {output_file}.')

# import json

# # 创建一个Python字典或列表，这里假设你有一个字典
# data = {"name": "Alice", "age": 30, "city": "New York"}

# # 将Python数据结构转换为JSON字符串
# json_string = json.dumps(data)

# # 指定要保存的JSON文件名
# json_filename = "output.json"

# # 将JSON字符串写入文件
# with open(json_filename, "w") as json_file:
#     json_file.write(json_string)

# print(f"数据已以字符串形式保存到 {json_filename} 文件中。")

