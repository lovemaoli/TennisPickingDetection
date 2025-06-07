"""
使用YOLO单例模式修改HybridTennisBallDetector类，避免重复下载模型
"""
import os
import time
import cv2
import numpy as np
import json
import warnings
import ssl
from pathlib import Path
import sys

# 尝试导入单例模式的YOLO检测器
try:
    from yolo_singleton import YOLOTennisBallDetector
    YOLO_SINGLETON_AVAILABLE = True
    print("成功导入YOLO单例模式检测器")
except ImportError:
    YOLO_SINGLETON_AVAILABLE = False
    print("无法导入YOLO单例模式检测器，将回退到本地实现")

# 导入其他模块（如果本地导入失败，则直接导入函数）
try:
    from src.tennis_detection import TennisBallDetector
    USING_LOCAL_MODULES = True
except ImportError:
    USING_LOCAL_MODULES = False
    # 如果在评测环境中无法导入本地模块，则直接定义基础检测类
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


class HybridTennisBallDetector:
    """混合检测器，结合传统方法和深度学习方法"""
    
    def __init__(self, use_yolo=True, use_traditional=True):
        """初始化混合检测器
        
        参数:
            use_yolo: 是否尝试使用YOLO模型
            use_traditional: 是否使用传统方法
        """
        # 初始化传统检测器
        self.traditional_detector = TennisBallDetector() if use_traditional else None
        
        # 如果可用且启用，初始化YOLO检测器
        self.yolo_detector = None
        if use_yolo and YOLO_SINGLETON_AVAILABLE:
            try:
                self.yolo_detector = YOLOTennisBallDetector()
                print("使用YOLO单例模式检测器")
            except Exception as e:
                print(f"创建YOLO单例检测器时出错: {e}")
                self.yolo_detector = None
        
        # 确保至少有一个检测器可用
        if not self.traditional_detector and not self.yolo_detector:
            print("警告: 没有可用的检测器，将使用默认的传统检测器")
            self.traditional_detector = TennisBallDetector()
        
    def detect(self, image_path):
        """结合两种方法检测网球
        
        参数:
            image_path: 图片路径
            
        返回:
            列表，包含每个检测到的网球位置
        """
        yolo_results = []
        traditional_results = []
        
        # 尝试使用YOLO检测
        if self.yolo_detector:
            try:
                yolo_results = self.yolo_detector.detect(image_path)
                if yolo_results:  
                    print(f"YOLO检测到 {len(yolo_results)} 个网球")
            except Exception as e:
                print(f"YOLO检测过程中出错: {e}")
        
        # 如果YOLO结果不理想，尝试传统方法
        if not yolo_results and self.traditional_detector:
            try:
                traditional_results = self.traditional_detector.detect(image_path)
                if traditional_results:
                    print(f"传统方法检测到 {len(traditional_results)} 个网球")
            except Exception as e:
                print(f"传统检测方法出错: {e}")
        
        # 返回更好的结果
        if yolo_results and not traditional_results:
            return yolo_results
        elif traditional_results and not yolo_results:
            return traditional_results
        elif yolo_results and traditional_results:
            # 如果两种方法都有结果，返回检测数量更合理的结果
            # 通常网球数量不会太多，如果传统方法检测出很多，可能是误检
            if len(traditional_results) > 5 and len(yolo_results) < 5:
                return yolo_results
            else:
                return traditional_results
        else:
            # 两种方法都没有结果
            return []


def process_img(img_path):
    """处理单张图片，识别网球位置
    
    参数:
        img_path: 图片路径
        
    返回:
        列表，包含每个检测到的网球位置 [{"x": x, "y": y, "w": w, "h": h}, ...]
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(img_path):
            print(f"错误: 图片不存在 - {img_path}")
            return []
            
        # 检查文件是否为图片
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"错误: 无法读取图片 - {img_path}")
                return []
        except Exception as e:
            print(f"读取图片时出错: {e}")
            return []
            
        # 尝试使用混合方法检测（先尝试YOLO，如果失败再用传统方法）
        try:
            detector = HybridTennisBallDetector(use_yolo=True, use_traditional=True)
            results = detector.detect(img_path)
            
            # 转换NumPy类型为标准Python类型
            if results:
                converted_results = []
                for ball in results:
                    converted_ball = {
                        'x': int(ball['x']),
                        'y': int(ball['y']),
                        'w': int(ball['w']),
                        'h': int(ball['h'])
                    }
                    # 保留置信度如果存在
                    if 'confidence' in ball:
                        converted_ball['confidence'] = float(ball['confidence'])
                    
                    converted_results.append(converted_ball)
                return converted_results
            
        except Exception as e:
            print(f"使用混合检测方法失败: {e}")
            
        return []
    except Exception as e:
        print(f"处理图片时出现未预期的错误: {e}")
        # 在出错的情况下，至少尝试返回一些结果
        try:
            return TennisBallDetector().detect(img_path)
        except:
            return []


#
# 以下代码仅作为测试时使用
#
if __name__=='__main__':
    imgs_folder = './imgs/'
    
    # 确认图片目录存在
    if not os.path.exists(imgs_folder):
        print(f"图片目录不存在: {imgs_folder}")
        print("尝试查找图片目录...")
        # 尝试在当前目录和父目录查找图片文件夹
        for root_dir in ['.', '..']:
            for dir_name in os.listdir(root_dir):
                if "赛题" in dir_name and "网球" in dir_name and os.path.isdir(os.path.join(root_dir, dir_name)):
                    imgs_folder = os.path.join(root_dir, dir_name)
                    print(f"找到图片目录: {imgs_folder}")
                    break
    
    # 获取所有图片文件
    img_paths = [f for f in os.listdir(imgs_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not img_paths:
        print(f"在 {imgs_folder} 中未找到图片文件")
        sys.exit(1)
        
    def now():
        return int(time.time()*1000)
    
    last_time = 0
    count_time = 0
    max_time = 0
    min_time = now()
    
    # 创建结果字典
    results_dict = {}
    
    for img_path in img_paths:
        full_path = os.path.join(imgs_folder, img_path)
        print(img_path, ':')
        last_time = now()
        result = process_img(full_path)
        run_time = now() - last_time
        
        # 保存结果
        results_dict[img_path] = result
        
        print('result:\n', result)
        print('run time: ', run_time, 'ms')
        print()
        count_time += run_time
        if run_time > max_time:
            max_time = run_time
        if run_time < min_time:
            min_time = run_time
    
    print('\n')
    print('avg time: ', int(count_time/len(img_paths)), 'ms')
    print('max time: ', max_time, 'ms')
    print('min time: ', min_time, 'ms')
    
    # 保存结果到文件
    output_file = 'detection_results.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=4, ensure_ascii=False)
    
    print(f"\n结果已保存到 {output_file}")
