import cv2
import os
import json
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from pathlib import Path

class ResultVisualizer:
    
    def __init__(self, image_folder, json_file):
        # 加载数据
        self.image_folder = image_folder
        self.json_file = json_file
        
        # 加载检测结果
        with open(json_file, 'r', encoding='utf-8') as f:
            self.detection_results = json.load(f)
        
        # 获取所有图像名称
        self.image_names = list(self.detection_results.keys())
        self.current_index = 0
        
        # 创建界面
        self.root = tk.Tk()
        self.root.title("网球检测结果可视化")
        self.root.geometry("1200x700")
        
        # 创建布局
        self.create_layout()
    
    def create_layout(self):
        # 创建顶部控制栏
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # 添加导航按钮
        self.prev_button = tk.Button(control_frame, text="上一张", command=self.show_prev)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.next_button = tk.Button(control_frame, text="下一张", command=self.show_next)
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        # 图像索引标签
        self.index_label = tk.Label(control_frame, text=f"图像: {self.current_index + 1}/{len(self.image_names)}")
        self.index_label.pack(side=tk.LEFT, padx=20)
        
        # 图像名称标签
        self.name_label = tk.Label(control_frame, text=f"文件名: {self.image_names[self.current_index] if self.image_names else 'N/A'}")
        self.name_label.pack(side=tk.LEFT, padx=20)
        
        # 检测结果信息
        self.result_label = tk.Label(control_frame, text="检测结果: 无")
        self.result_label.pack(side=tk.LEFT, padx=20)
        
        # 创建保存按钮
        self.save_button = tk.Button(control_frame, text="保存当前图像", command=self.save_current_image)
        self.save_button.pack(side=tk.RIGHT, padx=5)
        
        # 创建保存全部按钮
        self.save_all_button = tk.Button(control_frame, text="保存全部图像", command=self.save_all_images)
        self.save_all_button.pack(side=tk.RIGHT, padx=5)
        
        # 创建图像显示区域
        image_frame = tk.Frame(self.root)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 原始图像
        original_frame = tk.LabelFrame(image_frame, text="原始图像")
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.original_canvas = tk.Canvas(original_frame, bg="gray")
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 检测结果图像
        result_frame = tk.LabelFrame(image_frame, text="检测结果")
        result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.result_canvas = tk.Canvas(result_frame, bg="gray")
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 添加底部状态栏
        status_frame = tk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        self.status_label = tk.Label(status_frame, text="就绪", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT)
        
        # 绑定键盘事件
        self.root.bind("<Left>", lambda event: self.show_prev())
        self.root.bind("<Right>", lambda event: self.show_next())
        self.root.bind("<space>", lambda event: self.toggle_detection())
        
        # 显示第一张图片
        self.load_current_image()
    
    def load_current_image(self):
        if not self.image_names:
            self.status_label.config(text="没有图像可以显示")
            return
        
        # 获取当前图像名称
        image_name = self.image_names[self.current_index]
        image_path = os.path.join(self.image_folder, image_name)
        
        # 检查图像是否存在
        if not os.path.exists(image_path):
            self.status_label.config(text=f"图像不存在: {image_path}")
            return
        
        try:
            # 读取原始图像
            original_image = cv2.imread(image_path)
            if original_image is None:
                self.status_label.config(text=f"无法读取图像: {image_path}")
                return
            
            # 转换为RGB（OpenCV默认为BGR）
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # 创建检测结果图像的副本
            result_image = original_image.copy()
            
            # 获取检测结果
            boxes = self.detection_results.get(image_name, [])
            
            # 绘制检测框
            for i, box in enumerate(boxes):
                x, y, w, h = box['x'], box['y'], box['w'], box['h']
                
                # 绘制矩形框
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # 添加标签
                cv2.putText(result_image, f"Ball {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 在图片上添加总数信息
            ball_count = len(boxes)
            text = f"ball : {ball_count}"
            cv2.putText(result_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 转换检测结果图像为RGB
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            # 调整图像大小以适应画布
            original_image_rgb = self.resize_image_to_fit(original_image_rgb)
            result_image_rgb = self.resize_image_to_fit(result_image_rgb)
            
            # 将NumPy数组转换为PIL图像，然后转换为Tkinter可显示的格式
            self.original_photo = ImageTk.PhotoImage(image=Image.fromarray(original_image_rgb))
            self.result_photo = ImageTk.PhotoImage(image=Image.fromarray(result_image_rgb))
            
            # 更新画布
            self.original_canvas.config(width=self.original_photo.width(), height=self.original_photo.height())
            self.original_canvas.create_image(0, 0, anchor=tk.NW, image=self.original_photo)
            
            self.result_canvas.config(width=self.result_photo.width(), height=self.result_photo.height())
            self.result_canvas.create_image(0, 0, anchor=tk.NW, image=self.result_photo)
            
            # 更新标签信息
            self.index_label.config(text=f"图像: {self.current_index + 1}/{len(self.image_names)}")
            self.name_label.config(text=f"文件名: {image_name}")
            self.result_label.config(text=f"检测结果: {ball_count} 个网球")
            
            # 更新状态
            self.status_label.config(text=f"已加载图像: {image_name}")
            
            # 保存当前数据供保存使用
            self.current_original_image = original_image
            self.current_result_image = result_image
            
        except Exception as e:
            self.status_label.config(text=f"处理图像时出错: {e}")
    
    def resize_image_to_fit(self, image, max_height=600):
        # 获取原始尺寸
        h, w = image.shape[:2]
        
        # 计算缩放比例
        scale = min(1.0, max_height / h)
        
        # 如果图像已经足够小，则不需要缩放
        if scale >= 1.0:
            return image
        
        # 缩放图像
        new_h = int(h * scale)
        new_w = int(w * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def show_prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
    
    def show_next(self):
        if self.current_index < len(self.image_names) - 1:
            self.current_index += 1
            self.load_current_image()
    
    def toggle_detection(self):
        # 后续实现
        pass
    
    def save_current_image(self):
        if not hasattr(self, 'current_result_image'):
            self.status_label.config(text="没有图像可保存")
            return
        
        # 打开保存对话框
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")],
            initialfile=f"detected_{self.image_names[self.current_index]}"
        )
        
        if file_path:
            try:
                # 保存图像
                cv2.imwrite(file_path, self.current_result_image)
                self.status_label.config(text=f"图像已保存: {file_path}")
            except Exception as e:
                self.status_label.config(text=f"保存图像时出错: {e}")
    
    def save_all_images(self):
        # 打开文件夹选择对话框
        folder_path = filedialog.askdirectory(title="选择保存目录")
        
        if folder_path:
            # 确保目录存在
            os.makedirs(folder_path, exist_ok=True)
            
            # 保存原始索引
            original_index = self.current_index
            
            try:
                # 遍历所有图像并保存
                for i, image_name in enumerate(self.image_names):
                    # 更新当前索引并加载图像
                    self.current_index = i
                    self.load_current_image()
                    
                    # 保存检测结果图像
                    output_path = os.path.join(folder_path, f"detected_{image_name}")
                    cv2.imwrite(output_path, self.current_result_image)
                    
                    # 更新状态
                    self.status_label.config(text=f"处理: {i+1}/{len(self.image_names)} - {image_name}")
                    self.root.update()  # 更新界面
                
                # 恢复原始索引
                self.current_index = original_index
                self.load_current_image()
                
                self.status_label.config(text=f"已保存 {len(self.image_names)} 张图像到 {folder_path}")
                
            except Exception as e:
                self.status_label.config(text=f"保存图像时出错: {e}")
    
    def run(self):
        self.root.mainloop()


def main():
    # 获取项目根目录
    project_root = Path.cwd()
    
    # 查找图像文件夹和检测结果JSON文件
    image_folder = project_root / "imgs"
    json_file = project_root / "detection_results.json"
    
    # 检查路径是否存在
    if not image_folder.exists():
        for item in project_root.iterdir():
            if item.is_dir() and item.name.lower() in ["imgs", "images", "图片"]:
                image_folder = item
                break
        if not image_folder.exists():
            print(f"找不到图像文件夹")
            return
    
    if not json_file.exists():
        for item in project_root.iterdir():
            if item.is_file() and item.name.endswith(".json"):
                json_file = item
                break
        if not json_file.exists():
            print(f"找不到检测结果JSON文件")
            return
    
    # 打印路径信息
    print(f"图像文件夹: {image_folder}")
    print(f"JSON文件: {json_file}")
    
    # 创建并运行可视化器
    visualizer = ResultVisualizer(str(image_folder), str(json_file))
    visualizer.run()


if __name__ == "__main__":
    main()
