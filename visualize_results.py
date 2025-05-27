import cv2
import os
import json
from pathlib import Path

def draw_detection_results(image_folder, json_file, output_folder=None):
    """
    将检测结果绘制到图像上
    
    参数:
        image_folder: 图像文件夹
        json_file: 包含检测结果的JSON文件
        output_folder: 输出文件夹，如果不提供，默认为"output_images"
    """
    # 设置输出文件夹
    if output_folder is None:
        output_folder = "output_images"
    
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 加载检测结果
    with open(json_file, 'r', encoding='utf-8') as f:
        detection_results = json.load(f)
    
    # 遍历检测结果中的每个图片
    total_images = len(detection_results)
    processed_images = 0
    
    for image_name, boxes in detection_results.items():
        # 构建图像路径
        image_path = os.path.join(image_folder, image_name)
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"图像不存在: {image_path}")
            continue
        
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                continue
            
            # 绘制检测框
            for i, box in enumerate(boxes):
                x, y, w, h = box['x'], box['y'], box['w'], box['h']
                
                # 绘制矩形框
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # 添加标签
                cv2.putText(image, f"Ball {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 在图片上添加总数信息
            ball_count = len(boxes)
            text = f"检测到 {ball_count} 个网球"
            cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 保存结果图像
            output_path = os.path.join(output_folder, f"detected_{image_name}")
            cv2.imwrite(output_path, image)
            
            processed_images += 1
            print(f"处理进度: {processed_images}/{total_images} - {image_name}")
            
        except Exception as e:
            print(f"处理图像 {image_name} 时出错: {e}")
    
    print(f"已完成! 处理了 {processed_images}/{total_images} 张图像")


def main():
    # 获取项目根目录和相关路径
    project_root = Path.cwd()
    image_folder = project_root / "imgs"
    json_file = project_root / "detection_results.json"
    output_folder = project_root / "output_images"
    
    # 检查必要的路径是否存在
    if not image_folder.exists():
        # 尝试查找imgs文件夹
        for item in project_root.iterdir():
            if item.is_dir() and item.name.lower() in ["imgs", "images", "图片"]:
                image_folder = item
                break
    
    if not json_file.exists():
        print(f"找不到结果文件: {json_file}")
        return
    
    # 验证路径
    print(f"图像文件夹: {image_folder}")
    print(f"JSON文件: {json_file}")
    print(f"输出文件夹: {output_folder}")
    
    # 绘制检测结果
    draw_detection_results(str(image_folder), str(json_file), str(output_folder))


if __name__ == "__main__":
    main()
