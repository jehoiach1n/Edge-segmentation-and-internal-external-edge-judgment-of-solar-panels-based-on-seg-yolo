import json
import cv2
import numpy as np
import os

def generate_mask(image_path, json_path, output_dir):
    """
    生成单张图像的mask掩码
    :param image_path: 图像文件路径
    :param json_path: 对应的JSON标注文件路径
    :param output_dir: mask输出目录
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"警告：无法读取图像 {image_path}")
        return
    
    h, w = image.shape[:2]  # 获取图像尺寸
    
    # 创建空白mask（单通道，初始全0）
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 定义类别与像素值的映射（根据需求自定义）
    label_to_value = {
        "background": 0,
        "border_outer": 128,    # 灰色
        "border_inner": 255     # 白色
    }
    
    # 读取并解析json文件
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"处理JSON文件 {json_path} 时出错: {e}")
        return
    
    # 遍历每个标注对象，绘制到mask上
    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label not in label_to_value:
            print(f"警告：未知标签 {label}，已跳过")
            continue
            
        points = shape.get("points")  # 多边形坐标列表
        if not points:
            continue
            
        # 将坐标转为整数并重塑为OpenCV需要的格式
        try:
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            
            # 根据形状类型选择绘制函数
            shape_type = shape.get("shape_type", "polygon")
            if shape_type == "rectangle":
                # 矩形需要特殊处理，取对角点
                x1, y1 = pts[0][0][0], pts[0][0][1]
                x2, y2 = pts[1][0][0], pts[1][0][1]
                cv2.rectangle(mask, (x1, y1), (x2, y2), label_to_value[label], -1)
            else:  # 默认多边形
                cv2.fillPoly(mask, [pts], label_to_value[label])
        except Exception as e:
            print(f"绘制标注时出错: {e}")
            continue
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件名（与原图同名，添加_mask后缀）
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_mask{ext}")
    
    # 保存mask
    cv2.imwrite(output_path, mask)
    print(f"已生成mask: {output_path}")

def batch_process_images(image_dir, json_dir, output_dir):
    """
    批量处理文件夹中的所有图像和对应的JSON文件
    :param image_dir: 图像文件夹路径
    :param json_dir: JSON标注文件夹路径
    :param output_dir: mask输出文件夹路径
    """
    # 获取所有图像文件
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']
    image_files = [f for f in os.listdir(image_dir) 
                  if os.path.splitext(f)[1].lower() in image_extensions]
    
    for image_file in image_files:
        # 构建图像路径
        image_path = os.path.join(image_dir, image_file)
        
        # 构建对应的JSON文件路径（假设JSON与图像同名，仅扩展名不同）
        name, _ = os.path.splitext(image_file)
        json_file = f"{name}.json"
        json_path = os.path.join(json_dir, json_file)
        
        # 检查JSON文件是否存在
        if not os.path.exists(json_path):
            print(f"警告：未找到 {image_file} 对应的JSON文件 {json_file}，已跳过")
            continue
        
        # 生成mask
        generate_mask(image_path, json_path, output_dir)

if __name__ == "__main__":
    # 设置文件夹路径
    image_dir = "G:\\ai\\new_try\\all_dataset\\images"  # 图像所在文件夹
    json_dir = "new_try\\all_dataset\\labels"  # JSON标注所在文件夹
    output_dir = "G:\\ai\\new_try\\all_dataset\\masks"  # 生成的mask保存文件夹
    
    # 执行批量处理
    batch_process_images(image_dir, json_dir, output_dir)
    print("批量处理完成！")
