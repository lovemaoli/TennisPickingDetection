import cv2
import numpy as np
import os
import json
from pathlib import Path


class TennisBallDetector:
    """网球检测类，用于识别图片中的网球"""
    
    def __init__(self, hsv_lower=(25, 100, 100), hsv_upper=(65, 255, 255)):
        """初始化网球检测器
        
        参数:
            hsv_lower: HSV颜色空间的下界，默认为黄绿色的下界
            hsv_upper: HSV颜色空间的上界，默认为黄绿色的上界
        """
        self.hsv_lower = np.array(hsv_lower)
        self.hsv_upper = np.array(hsv_upper)
        
    def detect(self, image_path):
        """检测图片中的网球
        
        参数:
            image_path: 图片路径
            
        返回:
            列表，包含每个检测到的网球位置 [{'x': x, 'y': y, 'w': w, 'h': h}, ...]
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            return []
            
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 使用颜色阈值筛选网球区域
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # 应用形态学操作去除噪声
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 准备结果
        results = []
        
        # 分析每个轮廓
        for contour in contours:
            # 获取轮廓面积
            area = cv2.contourArea(contour)
            
            # 过滤太小的轮廓
            if area < 20:
                continue
                
            # 获取外接矩形
            x, y, w, h = cv2.boundingRect(contour)
            
            # 过滤非正方形区域（网球应该是圆形，所以宽高比应该接近1）
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.7 or aspect_ratio > 1.3:
                continue
                
            # 添加结果
            results.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h
            })
            
        return results
        
    def draw_results(self, image_path, results, output_path=None):
        """在图片上标记检测结果
        
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


def detect_all_images(input_dir, output_json=None, output_dir=None):
    """检测目录中所有图片的网球
    
    参数:
        input_dir: 输入图片目录
        output_json: 输出JSON文件路径
        output_dir: 输出标记后图片的目录
    """
    detector = TennisBallDetector()
    results_dict = {}
    
    # 确保输出目录存在
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
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
            
    return results_dict


def main():
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    
    # 设置输入和输出路径
    input_dir = project_root / "赛题3 - 智能捡网球机器人识别 - 测试图片及结果"
    output_json = project_root / "results" / "detection_results.json"
    output_dir = project_root / "results" / "marked_images"
    
    # 确保输出目录存在
    os.makedirs(output_dir.parent, exist_ok=True)
    
    # 处理所有图片
    results = detect_all_images(input_dir, output_json, output_dir)
    print(f"检测完成，共处理 {len(results)} 张图片")


if __name__ == "__main__":
    main()
