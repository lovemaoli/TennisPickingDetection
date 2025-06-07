import cv2
import numpy as np
import os
import json
import torch
import time
from pathlib import Path
from tennis_detection import TennisBallDetector
from yolo_detection import YOLOTennisBallDetector


class HybridTennisBallDetector:
    """混合网球检测器，结合传统方法和YOLO方法"""
      def __init__(self, model_path=None, hsv_lower=(25, 100, 100), hsv_upper=(65, 255, 255)):
        """
        初始化混合网球检测器
        
        参数:
            model_path: YOLO模型路径，如果为None，则使用YOLOv5s预训练模型
            hsv_lower: HSV颜色空间的下界，默认为黄绿色的下界
            hsv_upper: HSV颜色空间的上界，默认为黄绿色的上界
        """
        # 初始化两种检测器
        self.trad_detector = TennisBallDetector(hsv_lower, hsv_upper)
        self.yolo_detector = YOLOTennisBallDetector(model_path)
        
    def detect(self, image_path):
        """
        检测图片中的网球，结合两种方法
        
        参数:
            image_path: 图片路径
            
        返回:
            列表，包含每个检测到的网球位置 [{'x': x, 'y': y, 'w': w, 'h': h}, ...]
        """
        start_time = time.time()
        
        # 使用YOLO检测
        yolo_results = self.yolo_detector.detect(image_path)
        
        # 如果YOLO检测失败或没有结果，使用传统方法
        trad_results = []
        if not yolo_results:
            print("使用传统方法检测")
            trad_results = self.trad_detector.detect(image_path)
        
        # 合并结果，去重
        results = self._merge_results(yolo_results, trad_results, image_path)
        
        elapsed_time = int((time.time() - start_time) * 1000)  # 毫秒
        print(f"总检测耗时: {elapsed_time} 毫秒")
        
        return results
        
    def _merge_results(self, yolo_results, trad_results, image_path):
        """
        合并两种方法的检测结果，去除重叠区域
        
        参数:
            yolo_results: YOLO检测结果
            trad_results: 传统方法检测结果
            image_path: 图片路径，用于获取图片尺寸
            
        返回:
            合并后的结果
        """
        # 先添加所有YOLO结果
        merged_results = yolo_results.copy()
        
        # 读取图片获取尺寸
        image = cv2.imread(image_path)
        if image is None:
            return merged_results
            
        img_height, img_width = image.shape[:2]
        
        # 检查每个传统检测结果
        for trad_ball in trad_results:
            is_duplicate = False
            
            # 检查是否与任何YOLO结果重叠
            for yolo_ball in yolo_results:
                # 计算重叠区域
                x_overlap = max(0, min(trad_ball['x'] + trad_ball['w'], yolo_ball['x'] + yolo_ball['w']) - 
                              max(trad_ball['x'], yolo_ball['x']))
                y_overlap = max(0, min(trad_ball['y'] + trad_ball['h'], yolo_ball['y'] + yolo_ball['h']) - 
                              max(trad_ball['y'], yolo_ball['y']))
                overlap_area = x_overlap * y_overlap
                
                # 计算重叠比例
                trad_area = trad_ball['w'] * trad_ball['h']
                yolo_area = yolo_ball['w'] * yolo_ball['h']
                overlap_ratio = overlap_area / min(trad_area, yolo_area)
                
                # 如果重叠比例超过阈值，认为是重复检测
                if overlap_ratio > 0.3:
                    is_duplicate = True
                    break
            
            # 如果不是重复检测，添加到结果中
            if not is_duplicate:
                merged_results.append(trad_ball)
        
        # 额外过滤：确保所有检测结果都在合理范围内
        filtered_results = []
        for ball in merged_results:
            # 检查边界框是否在图片范围内
            if (ball['x'] >= 0 and ball['y'] >= 0 and 
                ball['x'] + ball['w'] <= img_width and 
                ball['y'] + ball['h'] <= img_height and
                ball['w'] > 5 and ball['h'] > 5):  # 过滤掉太小的检测结果
                filtered_results.append(ball)
                
        return filtered_results
        
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
        # 使用YOLO检测器的绘图方法
        return self.yolo_detector.draw_results(image_path, results, output_path)


def detect_all_images(input_dir, output_json=None, output_dir=None, model_path=None):
    """
    使用混合方法检测目录中所有图片的网球
    
    参数:
        input_dir: 输入图片目录
        output_json: 输出JSON文件路径
        output_dir: 输出标记后图片的目录
        model_path: YOLO模型路径
    """
    # 初始化混合检测器
    detector = HybridTennisBallDetector(model_path)
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
        
        # 将NumPy类型转换为Python标准类型以便JSON序列化
        converted_balls = []
        for ball in balls:
            converted_ball = {
                'x': int(ball['x']),
                'y': int(ball['y']),
                'w': int(ball['w']),
                'h': int(ball['h'])
            }
            # 保留置信度如果存在
            if 'confidence' in ball:
                converted_ball['confidence'] = float(ball['confidence'])
            
            converted_balls.append(converted_ball)
        
        results_dict[image_file] = converted_balls
        
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
    """主函数，运行混合检测"""
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    
    # 设置输入和输出路径
    input_dir = project_root / "赛题3 - 智能捡网球机器人识别 - 测试图片及结果"
    output_json = project_root / "results" / "hybrid_detection_results.json"
    output_dir = project_root / "results" / "hybrid_marked_images"
    
    # 确保输出目录存在
    os.makedirs(output_dir.parent, exist_ok=True)
    
    # 使用混合方法检测所有图片
    results = detect_all_images(input_dir, output_json, output_dir)
    print(f"检测完成，共处理 {len(results)} 张图片")


if __name__ == "__main__":
    main()