import argparse
import os
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / 'src'))

from tennis_detection import detect_all_images as detect_traditional
from yolo_detection import detect_all_images as detect_yolo
from hybrid_detection import detect_all_images as detect_hybrid


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='网球检测系统')
    parser.add_argument('--method', type=str, default='hybrid', choices=['traditional', 'yolo', 'hybrid'],
                        help='检测方法：traditional（传统方法）, yolo（深度学习方法）, hybrid（混合方法）')
    parser.add_argument('--input', type=str, default=None, 
                        help='输入图片目录路径，默认为项目根目录下的赛题3目录')
    parser.add_argument('--output', type=str, default=None, 
                        help='输出结果目录路径，默认为项目根目录下的results目录')
    parser.add_argument('--model', type=str, default=None,
                        help='YOLO模型路径，如果不指定则使用预训练模型')
    parser.add_argument('--train', action='store_true', 
                        help='是否训练YOLO模型，需要提供标签文件')
    parser.add_argument('--labels', type=str, default=None,
                        help='训练YOLO模型所需的标签文件路径，仅在--train参数指定时需要')
    
    args = parser.parse_args()
    
    # 获取项目根目录
    project_root = Path(__file__).parent
    
    # 设置输入和输出路径
    if args.input is None:
        input_dir = project_root / "赛题3 - 智能捡网球机器人识别 - 测试图片及结果"
    else:
        input_dir = Path(args.input)
    
    if args.output is None:
        output_dir = project_root / "results"
    else:
        output_dir = Path(args.output)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置输出文件路径
    output_json = output_dir / f"{args.method}_detection_results.json"
    output_images_dir = output_dir / f"{args.method}_marked_images"
    
    # 如果需要训练YOLO模型
    if args.train:
        if not args.labels:
            print("错误：需要提供标签文件路径（--labels参数）用于训练YOLO模型")
            return
        
        from yolo_detection import train_yolo_model, prepare_training_data
        
        print("准备训练数据...")
        yaml_path = prepare_training_data(input_dir, args.labels, output_dir=output_dir / 'dataset')
        
        print("开始训练YOLO模型...")
        model_path = train_yolo_model(yaml_path)
        
        if model_path:
            print(f"模型训练完成，保存在: {model_path}")
            args.model = model_path
        else:
            print("模型训练失败")
            return
    
    # 根据指定方法进行检测
    print(f"使用 {args.method} 方法进行检测...")
    
    if args.method == 'traditional':
        results = detect_traditional(input_dir, output_json, output_images_dir)
    elif args.method == 'yolo':
        results = detect_yolo(input_dir, output_json, output_images_dir, args.model)
    else:  # hybrid
        results = detect_hybrid(input_dir, output_json, output_images_dir, args.model)
    
    print(f"检测完成，结果保存在: {output_json}")
    print(f"标记后的图片保存在: {output_images_dir}")


if __name__ == "__main__":
    main()
