import cv2
import numpy as np
import os
import json
import torch
import urllib.request
import time
import warnings
import ssl
from pathlib import Path

# 全局变量，记录模型加载状态
_yolo_model = None
_model_load_attempted = False

class YOLOTennisBallDetector:
    
    def __init__(self, model_path=None):
        """
        初始化YOLO网球检测器
        
        参数:
            model_path: YOLO模型路径，如果为None，则使用YOLOv5m预训练模型
        """
        global _yolo_model, _model_load_attempted
        
        self.model = None
        try:
            # 检查是否已经有加载好的模型实例
            if _yolo_model is not None:
                print("使用已加载的YOLO模型")
                self.model = _yolo_model
            # 如果之前尝试过加载但失败了，就不再尝试，避免重复报错
            elif _model_load_attempted:
                print("YOLO模型初始化失败，将只使用传统方法")
            else:
                _model_load_attempted = True
                
                # 配置SSL上下文，忽略证书验证问题
                ssl_context = ssl._create_unverified_context()
                
                # 设置更长的超时时间
                timeout = 30  # 30秒超时
                
                # 禁止警告
                warnings.filterwarnings("ignore", category=UserWarning)
                
                # 加载本地模型
                if model_path and os.path.exists(model_path):
                    print("加载本地YOLO模型")
                    self.model = torch.hub.load(
                        'ultralytics/yolov5', 
                        'custom', 
                        path=model_path, 
                        force_reload=False, 
                        verbose=False
                    )
                    print("YOLO模型加载成功，使用设备:", self.model.device)
                else:
                    # 使用本地保存的预训练模型
                    local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "yolov5m.pt")
                    if os.path.exists(local_path):
                        print("加载本地预训练模型:", local_path)
                        self.model = torch.hub.load(
                            'ultralytics/yolov5', 
                            'custom', 
                            path=local_path, 
                            force_reload=False, 
                            verbose=False
                        )
                    else:
                        # 下载预训练模型
                        print("加载预训练模型")
                        self.model = torch.hub.load(
                            'ultralytics/yolov5', 
                            'yolov5m', 
                            pretrained=True, 
                            force_reload=False, 
                            verbose=False
                        )
                        # 保存模型到本地以便下次使用
                        self.model.save(local_path)
                        
                    print("YOLO模型加载成功，使用设备:", self.model.device)
                
                # 设置为CPU模式，避免CUDA问题
                self.model.to('cpu')
                
                # 保存到全局变量
                _yolo_model = self.model
                
        except Exception as e:
            print(f"初始化YOLO模型失败，将使用传统方法: {str(e)}")
            self.model = None
        
        # 设置网球类别（YOLOv5m预训练模型中的第32类为"sports ball"）
        self.ball_class = 32
        
        # 设置置信度阈值
        if self.model:
            self.model.conf = 0.5
        
    def detect(self, image_path):
        """
        检测图片中的网球
        
        参数:
            image_path: 图片路径
            
        返回:
            列表，包含每个检测到的网球位置 [{'x': x, 'y': y, 'w': w, 'h': h}, ...]
        """
        # 如果模型加载失败，直接返回空列表
        if self.model is None:
            return []
            
        # 确保图片存在
        if not os.path.exists(image_path):
            print(f"无法找到图片: {image_path}")
            return []
        
        balls = []
        start_time = time.time()
        
        try:
            # 推理
            results = self.model(image_path)
            
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
                        'h': h,
                        'confidence': float(conf)
                    })
            
            # 打印检测结果
            if len(balls) > 0:
                print(f"YOLO检测到 {len(balls)} 个网球")
                
        except Exception as e:
            print(f"YOLO检测失败: {str(e)}")
            
        elapsed_time = int((time.time() - start_time) * 1000)  # 毫秒
        print(f"YOLO检测耗时: {elapsed_time} 毫秒")
                
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
            
            # 如果有置信度，显示置信度
            if 'confidence' in ball:
                conf_text = f"{ball['confidence']:.2f}"
                cv2.putText(image, conf_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        # 保存结果
        if output_path:
            cv2.imwrite(output_path, image)
            
        return image
