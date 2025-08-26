import os

# 要处理的文件夹
folder_path = 'G:\\ai\\new_labelme_json'

# 存储统一后的路径
fixed_paths = []  

# 遍历文件夹
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # 获取完整路径
        original_path = os.path.join(root, file)  
        # 替换为正斜杠（兼容跨平台）
        fixed_path = original_path.replace("\\", "/")  
        fixed_paths.append(fixed_path)

# 打印结果（也可以写入文件）
for path in fixed_paths:
    print(path)

print(f"\n共处理 {len(fixed_paths)} 个文件，路径已统一为正斜杠风格")