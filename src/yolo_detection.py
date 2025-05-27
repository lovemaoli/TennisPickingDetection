import cv2
import numpy as np
import os
import json
import torch
from pathlib import Path


class YOLOTennisBallDetector:
    """使用YOLO模型的网球检测器"""
    
    def __init__(self, model_path=None):
        """
        初始化YOLO网球检测器
        
        参数:
            model_path: YOLO模型路径，如果为None，则使用YOLOv5s预训练模型
        """
        # 加载YOLO模型
        if model_path and os.path.exists(model_path):
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        else:
            # 使用预训练模型
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            
        # 设置网球类别（YOLOv5s预训练模型中的第32类为"sports ball"）
        self.ball_class = 32
        
        # 设置置信度阈值
        self.model.conf = 0.5
        
    def detect(self, image_path):
        """
        检测图片中的网球
        
        参数:
            image_path: 图片路径
            
        返回:
            列表，包含每个检测到的网球位置 [{'x': x, 'y': y, 'w': w, 'h': h}, ...]
        """
        # 确保图片存在
        if not os.path.exists(image_path):
            print(f"无法找到图片: {image_path}")
            return []
            
        # 推理
        results = self.model(image_path)
        
        # 提取网球检测结果
        balls = []
        
        # 处理检测结果
        for detection in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
            
            # 只保留网球类别或所有类别（如果使用自定义模型）
            if self.model.names[int(cls)] == 'sports ball' or int(cls) == self.ball_class:
                # 计算边界框
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                
                balls.append({
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h
                })
                
        return balls
        
    def draw_results(self, image_path, results, output_path=None):
        """
        在图片上标记检测结果
        
        参数:
            image_path: 原图路径
            results: 检测结果
            output_path: 输出图片路径，如果为None，则不保存
            
        返回:
            带标记的图片
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            return None
            
        # 绘制检测结果
        for ball in results:
            x, y, w, h = ball['x'], ball['y'], ball['w'], ball['h']
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        # 保存结果
        if output_path:
            cv2.imwrite(output_path, image)
            
        return image


def train_yolo_model(train_data_path, epochs=50, img_size=640, batch_size=16):
    """
    训练YOLO模型
    
    参数:
        train_data_path: 训练数据YAML文件路径
        epochs: 训练轮数
        img_size: 图片大小
        batch_size: 批次大小
    
    返回:
        训练好的模型路径
    """
    # 克隆YOLOv5仓库（如果尚未克隆）
    import subprocess
    import sys
    
    if not os.path.exists('yolov5'):
        print("克隆YOLOv5仓库...")
        try:
            subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5.git'], 
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"克隆YOLOv5仓库失败: {e}")
            return None
        
    # 安装依赖
    print("安装依赖...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'yolov5/requirements.txt'], 
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"安装依赖失败: {e}")
        return None
    
    # 训练模型
    print(f"开始训练模型，共{epochs}轮...")
    try:
        # 使用跨平台方式运行训练命令
        cmd = [
            sys.executable, 
            'yolov5/train.py', 
            '--data', train_data_path, 
            '--epochs', str(epochs), 
            '--img', str(img_size), 
            '--batch', str(batch_size), 
            '--name', 'tennis_ball_model'
        ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"训练模型失败: {e}")
        return None
    
    # 返回训练好的模型路径
    model_path = os.path.join('yolov5', 'runs', 'train', 'tennis_ball_model', 'weights', 'best.pt')
    if os.path.exists(model_path):
        return model_path
    else:
        print("训练完成但未找到模型文件")
        return None


def prepare_training_data(data_dir, labels_file, output_dir='dataset'):
    """
    准备YOLO训练数据
    
    参数:
        data_dir: 图片目录
        labels_file: 标签文件(JSON格式)
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(f'{output_dir}/images/train', exist_ok=True)
    os.makedirs(f'{output_dir}/images/val', exist_ok=True)
    os.makedirs(f'{output_dir}/labels/train', exist_ok=True)
    os.makedirs(f'{output_dir}/labels/val', exist_ok=True)
    
    # 加载标签
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    # 获取所有图片
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # 划分训练集和验证集
    np.random.shuffle(image_files)
    split = int(len(image_files) * 0.8)
    train_files = image_files[:split]
    val_files = image_files[split:]
      # 处理训练集
    for img_file in train_files:
        # 复制图片
        src_img = os.path.join(data_dir, img_file)
        dst_img = os.path.join(output_dir, 'images', 'train', img_file)
        # 使用跨平台的方式复制文件
        import shutil
        try:
            shutil.copy2(src_img, dst_img)
        except Exception as e:
            print(f"复制文件失败: {e}")
        
        # 创建标签文件
        if img_file in labels:
            create_yolo_label(labels[img_file], os.path.join(output_dir, 'labels', 'train', os.path.splitext(img_file)[0] + '.txt'))
    
    # 处理验证集
    for img_file in val_files:
        # 复制图片
        src_img = os.path.join(data_dir, img_file)
        dst_img = os.path.join(output_dir, 'images', 'val', img_file)
        # 使用跨平台的方式复制文件
        try:
            shutil.copy2(src_img, dst_img)
        except Exception as e:
            print(f"复制文件失败: {e}")
        
        # 创建标签文件
        if img_file in labels:
            create_yolo_label(labels[img_file], os.path.join(output_dir, 'labels', 'val', os.path.splitext(img_file)[0] + '.txt'))
            
    # 创建数据集配置文件
    with open(f'{output_dir}/tennis.yaml', 'w') as f:
        f.write(f'''path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
nc: 1
names: ['tennis_ball']''')
            
    return f'{output_dir}/tennis.yaml'


def create_yolo_label(balls, output_file):
    """
    创建YOLO格式的标签文件
    
    参数:
        balls: 网球位置列表
        output_file: 输出文件路径
    """
    # 获取图片大小（这里假设为640x640，实际应该从图片获取）
    img_width, img_height = 640, 640
    
    with open(output_file, 'w') as f:
        for ball in balls:
            # 计算YOLO格式的坐标（归一化的中心点坐标和宽高）
            x_center = (ball['x'] + ball['w'] / 2) / img_width
            y_center = (ball['y'] + ball['h'] / 2) / img_height
            width = ball['w'] / img_width
            height = ball['h'] / img_height
            
            # 写入文件（类别为0，表示网球）
            f.write(f"0 {x_center} {y_center} {width} {height}\n")


def main():
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    
    # 设置输入和输出路径
    input_dir = project_root / "赛题3 - 智能捡网球机器人识别 - 测试图片及结果"
    output_json = project_root / "results" / "detection_results.json"
    output_dir = project_root / "results" / "marked_images"
    
    # 确保输出目录存在
    os.makedirs(output_dir.parent, exist_ok=True)
    
    # 使用预训练模型进行检测
    detector = YOLOTennisBallDetector()
    results_dict = {}
    
    # 处理所有图片
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        print(f"处理图片: {image_file}")
        
        # 检测网球
        balls = detector.detect(image_path)
        results_dict[image_file] = balls
        
        # 保存标记后的图片
        if output_dir:
            output_path = os.path.join(output_dir, f"marked_{image_file}")
            detector.draw_results(image_path, balls, output_path)
    
    # 保存检测结果
    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=4, ensure_ascii=False)
            
    print(f"检测完成，共处理 {len(results_dict)} 张图片")


if __name__ == "__main__":
    main()
