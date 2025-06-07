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
        self.min_radius = 10  # 网球半径最小值（像素）
        self.max_radius = 60  # 网球半径最大值（像素）
        
    def enhance_image(self, image):
        """增强图像，提高网球检测效果
        
        参数:
            image: 原始图像
            
        返回:
            增强后的图像
        """
        # 均衡化亮度通道，保持颜色不变
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 轻微锐化以突出边缘
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
    def detect_circles(self, image):
        """使用Hough圆变换检测网球
        
        参数:
            image: 原始图像
            
        返回:
            检测到的圆形列表
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 中值滤波减少噪声，保留边缘
        blurred = cv2.medianBlur(gray, 5)
        
        # 使用Hough圆变换检测圆
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=30,  # 圆心之间的最小距离
            param1=50,   # Canny边缘检测的高阈值
            param2=30,   # 圆心检测的累加器阈值
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        results = []
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # 获取圆的参数（防止溢出）
                center_x, center_y, radius = int(i[0]), int(i[1]), int(i[2])
                x = max(0, center_x - radius)  # 安全地计算左上角x坐标
                y = max(0, center_y - radius)  # 安全地计算左上角y坐标
                w = h = radius * 2  # 宽度和高度
                
                # 确保边界不超出图像范围
                if x + w > image.shape[1]:
                    w = image.shape[1] - x
                if y + h > image.shape[0]:
                    h = image.shape[0] - y
                
                # 检查是否为网球颜色（黄绿色）
                # 提取圆形区域
                if w <= 0 or h <= 0:  # 检查尺寸是否有效
                    continue
                    
                roi = image[y:y+h, x:x+w]
                if roi.size == 0:  # 检查ROI是否有效
                    continue
                
                # 转换到HSV检查颜色
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_roi, self.hsv_lower, self.hsv_upper)
                green_ratio = cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1])
                
                # 只选择具有足够黄绿色比例的圆
                if green_ratio > 0.3:
                    results.append({
                        'x': max(0, x),
                        'y': max(0, y),
                        'w': w,
                        'h': h,
                        'confidence': green_ratio
                    })
        
        return results
    
    def split_connected_balls(self, mask, min_dist=30):
        """分离连接在一起的网球
        
        参数:
            mask: 二值掩码图像
            min_dist: 两个球心之间的最小距离
            
        返回:
            分离后的标记列表 [{'x': x, 'y': y, 'w': w, 'h': h}, ...]
        """
        # 使用距离变换找到可能的球心位置
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # 归一化距离变换结果
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        
        # 寻找局部最大值作为种子点
        dist_transform_8u = (dist_transform * 255).astype(np.uint8)
        _, maxima = cv2.threshold(dist_transform_8u, 0.5 * dist_transform_8u.max(), 255, cv2.THRESH_BINARY)
        
        # 使用形态学运算找到分水岭标记
        kernel = np.ones((3, 3), np.uint8)
        markers = cv2.dilate(maxima, kernel)
        
        # 连通区域分析找到所有可能的球心
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(markers, connectivity=8)
        
        results = []
        
        # 从stats中提取边界框
        for i in range(1, num_labels):  # 从1开始，跳过背景
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            # 过滤太小的区域
            if area < 100:
                continue
                
            # 估计半径
            radius = max(w, h) // 2
            
            # 网球应该是大致圆形
            if abs(w - h) > 10 or radius < self.min_radius or radius > self.max_radius:
                continue
                
            # 添加到结果
            results.append({
                'x': max(0, x - radius // 2),  # 扩大边界框以确保包含整个球
                'y': max(0, y - radius // 2),
                'w': w + radius,
                'h': h + radius
            })
            
        return results
        
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
        
        # 图片预处理 - 增强对比度，便于更好地检测
        image_enhanced = self.enhance_image(image)
            
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(image_enhanced, cv2.COLOR_BGR2HSV)
        
        # 使用颜色阈值筛选网球区域
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # 应用形态学操作去除噪声
        open_kernel = np.ones((3, 3), np.uint8)  # 使用较小的核先开操作
        close_kernel = np.ones((7, 7), np.uint8)  # 使用较大的核闭操作，连接相邻区域
        
        # 先开操作去噪点
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        # 再闭操作连接相邻区域
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        
        # 首先尝试使用Hough圆检测，这通常能更好地分辨相邻的球
        circle_results = self.detect_circles(image_enhanced)
        
        # 当检测到的圆少于10个并且大于0个时，使用圆检测结果
        if len(circle_results) > 0 and len(circle_results) < 10:
            # 过滤重叠的圆
            filtered_circles = []
            for i, circle1 in enumerate(circle_results):
                is_overlapped = False
                for j, circle2 in enumerate(filtered_circles):
                    # 计算中心点（确保使用 int 类型避免溢出）
                    c1_x = int(circle1['x']) + int(circle1['w']) // 2
                    c1_y = int(circle1['y']) + int(circle1['h']) // 2
                    c2_x = int(circle2['x']) + int(circle2['w']) // 2
                    c2_y = int(circle2['y']) + int(circle2['h']) // 2
                    
                    # 计算距离（使用float以避免溢出）
                    dist = np.sqrt(float((c1_x - c2_x) ** 2) + float((c1_y - c2_y) ** 2))
                    
                    # 如果距离小于两个圆半径之和的一半，认为是重叠的
                    if dist < (circle1['w'] + circle2['w']) / 4:
                        is_overlapped = True
                        # 保留置信度更高的圆
                        if circle1.get('confidence', 0) > circle2.get('confidence', 0):
                            filtered_circles[j] = circle1
                        break
                
                if not is_overlapped:
                    filtered_circles.append(circle1)
                        
            # 使用过滤后的圆形检测结果
            return filtered_circles
        
        # 如果Hough圆检测结果不理想，尝试处理连接在一起的网球
        # 尝试分离连接的网球
        connected_balls = self.split_connected_balls(mask)
        if len(connected_balls) > 0:
            # 验证分离出的球的颜色
            valid_balls = []
            for ball in connected_balls:
                x, y, w, h = ball['x'], ball['y'], ball['w'], ball['h']
                
                # 限制在图像边界内
                img_h, img_w = image.shape[:2]
                x = max(0, min(x, img_w - 1))
                y = max(0, min(y, img_h - 1))
                w = min(w, img_w - x)
                h = min(h, img_h - y)
                
                if w <= 0 or h <= 0:
                    continue
                
                # 检查颜色
                roi = image[y:y+h, x:x+w]
                if roi.size > 0:  # 确保ROI不为空
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    yellow_green_mask = cv2.inRange(hsv_roi, self.hsv_lower, self.hsv_upper)
                    yellow_green_ratio = cv2.countNonZero(yellow_green_mask) / (w * h)
                    
                    if yellow_green_ratio > 0.3:  # 黄绿色比例阈值
                        ball['confidence'] = yellow_green_ratio
                        valid_balls.append(ball)
            
            # 如果有有效的分离结果，返回
            if len(valid_balls) > 0:
                return valid_balls
        
        # 寻找轮廓作为备选方案
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 准备结果
        results = []
        
        # 分析每个轮廓
        for contour in contours:
            # 获取轮廓面积
            area = cv2.contourArea(contour)
            
            # 过滤太小的轮廓 (增加面积阈值，排除小噪点)
            min_area = 100  # 增大最小面积阈值，减少小绿色物体的误检
            if area < min_area:
                continue
                
            # 获取外接矩形
            x, y, w, h = cv2.boundingRect(contour)
            
            # 过滤形状异常的区域 (更严格的宽高比约束)
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.7 or aspect_ratio > 1.3:
                continue
                
            # 计算圆度 (圆的轮廓的圆度接近1)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # 网球应该是圆形，圆度应该接近1
            if circularity < 0.65:  # 提高圆度要求，减少非网球的误检
                continue
                
            # 边界检查：如果轮廓太接近图像边缘，可能不是完整的网球
            img_h, img_w = image.shape[:2]
            border = 5
            if x <= border or y <= border or x + w >= img_w - border or y + h >= img_h - border:
                continue
            
            # 检查颜色：网球通常是黄绿色，中心部分颜色应该更为集中
            roi = image[y:y+h, x:x+w]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # 计算ROI区域内的黄绿色比例
            yellow_green_mask = cv2.inRange(hsv_roi, self.hsv_lower, self.hsv_upper)
            yellow_green_ratio = cv2.countNonZero(yellow_green_mask) / (w * h)
            
            if yellow_green_ratio < 0.45:  # 提高黄绿色比例要求，减少非网球的误检
                continue
                
            # 如果是较大的区域，检查是否可能是多个连在一起的网球
            if w > self.max_radius * 1.8 and h > self.max_radius * 1.8:
                # 创建该轮廓的掩码
                contour_mask = np.zeros(mask.shape, np.uint8)
                cv2.drawContours(contour_mask, [contour], 0, 255, -1)
                
                # 尝试分离连接的球
                split_balls = self.split_connected_balls(contour_mask)
                
                # 如果成功分离出多个球，添加它们而不是原始的大轮廓
                if len(split_balls) > 1:
                    # 验证分离出的球
                    for ball in split_balls:
                        sx, sy, sw, sh = ball['x'], ball['y'], ball['w'], ball['h']
                        # 检查颜色
                        s_roi = image[sy:sy+sh, sx:sx+sw]
                        if s_roi.size > 0:
                            s_hsv_roi = cv2.cvtColor(s_roi, cv2.COLOR_BGR2HSV)
                            s_mask = cv2.inRange(s_hsv_roi, self.hsv_lower, self.hsv_upper)
                            s_ratio = cv2.countNonZero(s_mask) / (sw * sh)
                            
                            if s_ratio > 0.4:
                                ball['confidence'] = s_ratio
                                results.append(ball)
                    continue  # 跳过添加原始大轮廓
            
            # 添加结果
            results.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'confidence': circularity * yellow_green_ratio  # 添加置信度分数
            })
            
        # 去除重叠的结果
        filtered_results = []
        for i, r1 in enumerate(results):
            is_duplicate = False
            for j, r2 in enumerate(filtered_results):
                # 计算重叠区域
                x_overlap = max(0, min(r1['x'] + r1['w'], r2['x'] + r2['w']) - max(r1['x'], r2['x']))
                y_overlap = max(0, min(r1['y'] + r1['h'], r2['y'] + r2['h']) - max(r1['y'], r2['y']))
                overlap_area = x_overlap * y_overlap
                
                # 计算重叠比例
                area1 = r1['w'] * r1['h']
                area2 = r2['w'] * r2['h']
                overlap_ratio = overlap_area / min(area1, area2)
                
                if overlap_ratio > 0.5:  # 重叠超过50%认为是重复
                    is_duplicate = True
                    # 保留置信度更高的结果
                    if r1.get('confidence', 0) > r2.get('confidence', 0):
                        filtered_results[j] = r1
                    break
            
            if not is_duplicate:
                filtered_results.append(r1)
                
        return filtered_results
        
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
            detector.draw_results(image_path, converted_balls, output_path)
    
    # 保存检测结果
    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=4, ensure_ascii=False)
            
    return results_dict


def main():
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    
    # 设置输入和输出路径
    input_dir = project_root / "imgs"
    output_json = project_root / "detection_results.json"
    output_dir = project_root / "results" / "marked_images"
    
    # 确保输出目录存在
    os.makedirs(output_dir.parent, exist_ok=True)
    
    # 处理所有图片
    results = detect_all_images(input_dir, output_json, output_dir)
    print(f"检测完成，共处理 {len(results)} 张图片")


if __name__ == "__main__":
    main()
