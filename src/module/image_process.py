import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import time
from datetime import datetime
from PIL import Image, ImageOps
import torch
from torchvision import transforms

class ImageProcessModule:
    def __init__(self, output_dir=None):
        """初始化图像处理模块
        
        参数:
            output_dir: 输出目录，如果为None则自动创建时间戳目录
        """
        # 模型输入尺寸
        self.target_size = (28, 28)
        # 存储处理过程中的中间图像
        self.cropped_image = None
        self.resized_image = None
        self.binary_image = None  # 新增二值化图像存储
        self.normalized_image = None
        
        # 设置输出目录
        if output_dir is None:
            self.output_dir = self._create_timestamp_dir()
        else:
            self.output_dir = output_dir
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                
        # 设置归一化转换器，与detect.py保持一致
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def _create_timestamp_dir(self):
        """创建以时间戳命名的子目录
        
        返回:
            创建的目录路径
        """
        # 确保基础输出目录存在
        if not os.path.exists("output"):
            os.makedirs("output")
        
        # 创建时间戳子目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("output", timestamp)
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")
        
        return output_dir
    
    def process_image(self, image):
        """处理图像，包括裁剪、缩放、二值化和归一化
        
        参数:
            image: 输入图像，numpy数组格式
        
        返回:
            处理后的归一化图像，用于模型输入
        """
        # 智能裁剪
        self.cropped_image = self._smart_crop(image)
        
        # 缩放到标准大小
        self.resized_image = self._resize_image(self.cropped_image)
        
        # 二值化处理（与detect.py一致）
        self.binary_image = self._binarize_image(self.resized_image)
        
        # 归一化处理（使用与detect.py相同的归一化参数）
        self.normalized_image = self._normalize_image(self.binary_image)
        
        # 返回归一化后的图像
        return self.normalized_image
    
    def _smart_crop(self, image):
        """智能裁剪，根据笔迹范围确定最佳裁剪区域
        
        参数:
            image: 输入图像，numpy数组格式
        
        返回:
            裁剪后的图像
        """
        # 寻找笔迹区域
        # 为了性能，先降低分辨率
        small_img = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        
        # 找到非零像素位置
        y_indices, x_indices = np.where(small_img > 0)
        
        # 检查是否有笔迹
        if len(x_indices) == 0 or len(y_indices) == 0:
            print("未检测到笔迹，返回原图")
            return image
        
        # 计算笔迹边界
        min_x, max_x = np.min(x_indices), np.max(x_indices)
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        
        # 缩放回原始尺寸
        min_x, max_x = min_x * 4, max_x * 4
        min_y, max_y = min_y * 4, max_y * 4
        
        # 计算宽度和高度，并取最大值作为正方形边长
        width = max_x - min_x
        height = max_y - min_y
        size = int(max(width, height) * 1.5)  # 乘以1.1增加边距
        
        # 计算正方形裁剪区域的中心
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        
        # 计算正方形裁剪区域的左上角坐标
        x1 = max(0, center_x - size // 2)
        y1 = max(0, center_y - size // 2)
        
        # 确保裁剪区域不超出图像边界
        x2 = min(image.shape[1], x1 + size)
        y2 = min(image.shape[0], y1 + size)
        
        # 如果右边界或下边界超出图像，需要重新调整左边界或上边界
        if x2 == image.shape[1] and x2 - x1 < size:
            x1 = max(0, x2 - size)
        if y2 == image.shape[0] and y2 - y1 < size:
            y1 = max(0, y2 - size)
        
        # 裁剪图像
        cropped = image[y1:y2, x1:x2]
        
        # 打印裁剪信息
        print(f"智能裁剪: 从({x1},{y1})到({x2},{y2}), 大小: {cropped.shape}")
        
        return cropped
    
    def _resize_image(self, image):
        """将图像缩放到模型要求的尺寸
        
        参数:
            image: 输入图像
        
        返回:
            缩放后的图像
        """
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
    
    def _binarize_image(self, image):
        """二值化图像，与detect.py保持一致
        
        参数:
            image: 输入图像
        
        返回:
            二值化后的图像
        """
        # 确保图像像素值在0-255范围内
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        # 应用二值化，阈值为128
        _, binary = cv2.threshold(image, 60, 255, cv2.THRESH_BINARY)
        
        return binary
    
    def _normalize_image(self, image):
        """归一化图像，与detect.py保持一致
        
        参数:
            image: 输入图像
        
        返回:
            归一化后的图像
        """
        # 确保图像像素值在0-255范围内
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        # 转换为PIL图像以便使用和detect.py相同的处理流程
        pil_image = Image.fromarray(image)
        
        # # 反色
        # pil_image = ImageOps.invert(pil_image)
        
        # 应用相同的转换
        tensor_image = self.transform(pil_image)
        
        # 转回numpy数组，保持与原代码兼容
        normalized = tensor_image.numpy()[0]  # 移除通道维度
        
        return normalized
    
    def show_processing_steps(self):
        """显示图像处理的各个步骤"""
        if self.cropped_image is None or self.resized_image is None or self.normalized_image is None:
            print("尚未处理图像，无法显示处理步骤")
            return
        
        # 创建一个4列的子图布局
        fig = plt.figure(figsize=(20, 5))
        gs = GridSpec(1, 4, figure=fig)
        
        # 显示裁剪后的图像
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self.cropped_image, cmap='gray')
        ax1.set_title("Cropped Image")
        ax1.axis('off')
        
        # 显示缩放后的图像
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(self.resized_image, cmap='gray')
        ax2.set_title(f"Resized to {self.target_size}")
        ax2.axis('off')
        
        # 显示二值化后的图像
        ax3 = fig.add_subplot(gs[0, 2])
        if self.binary_image is not None:
            ax3.imshow(self.binary_image, cmap='gray')
            ax3.set_title("Binary Image")
        else:
            ax3.set_title("Binary Image (Not Available)")
        ax3.axis('off')
        
        # 显示归一化后的图像
        ax4 = fig.add_subplot(gs[0, 3])
        # 反转显示，因为模型输入是黑底白字
        ax4.imshow(self.normalized_image, cmap='gray')
        ax4.set_title("Normalized Image (Model Input)")
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
    
    def save_processed_image(self, filename_prefix="handwritten"):
        """保存处理后的图像
        
        参数:
            filename_prefix: 文件名前缀
        
        返回:
            保存的文件路径
        """
        if self.normalized_image is None:
            print("尚未处理图像，无法保存")
            return None
        
        # 保存归一化后的图像
        norm_filename = os.path.join(self.output_dir, f"{filename_prefix}_processed.png")
        
        # 保存为PIL图像格式
        norm_image = Image.fromarray((self.normalized_image * 255).astype(np.uint8))
        norm_image.save(norm_filename)
        
        print(f"处理后的图像已保存至 {norm_filename}")
        
        return norm_filename
    
    def save_all_processing_steps(self, filename_prefix="handwritten"):
        """保存所有处理阶段的图像
        
        参数:
            filename_prefix: 文件名前缀
        
        返回:
            包含所有保存文件路径的字典
        """
        if self.cropped_image is None or self.resized_image is None or self.normalized_image is None:
            print("尚未处理图像，无法保存")
            return None
        
        saved_files = {}
        
        # 保存裁剪后的图像
        cropped_filename = os.path.join(self.output_dir, f"{filename_prefix}_cropped.png")
        cv2.imwrite(cropped_filename, self.cropped_image)
        saved_files['cropped'] = cropped_filename
        print(f"裁剪后的图像已保存至 {cropped_filename}")
        
        # 保存缩放后的图像
        resized_filename = os.path.join(self.output_dir, f"{filename_prefix}_resized.png")
        cv2.imwrite(resized_filename, self.resized_image)
        saved_files['resized'] = resized_filename
        print(f"缩放后的图像已保存至 {resized_filename}")
        
        # 保存二值化后的图像
        if self.binary_image is not None:
            binary_filename = os.path.join(self.output_dir, f"{filename_prefix}_binary.png")
            cv2.imwrite(binary_filename, self.binary_image)
            saved_files['binary'] = binary_filename
            print(f"二值化后的图像已保存至 {binary_filename}")
        
        # 保存归一化后的图像（用于直接查看）
        norm_filename = os.path.join(self.output_dir, f"{filename_prefix}_normalized.png")
        # 转换为8位图像并保存
        norm_image = Image.fromarray((self.normalized_image * 255).astype(np.uint8))
        norm_image.save(norm_filename)
        saved_files['normalized'] = norm_filename
        print(f"归一化后的图像已保存至 {norm_filename}")
        
        # 保存可视化图像
        vis_filename = os.path.join(self.output_dir, f"{filename_prefix}_visualization.png")
        self._save_visualization(vis_filename)
        saved_files['visualization'] = vis_filename
        
        print(f"共保存了{len(saved_files)}个处理步骤的图像到 {self.output_dir}")
        return saved_files
    
    def _save_visualization(self, filename):
        """保存处理步骤的可视化图像
        
        参数:
            filename: 保存的文件名
        """
        # 创建图像
        fig = plt.figure(figsize=(20, 5))
        gs = GridSpec(1, 4, figure=fig)
        
        # 显示裁剪后的图像
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self.cropped_image, cmap='gray')
        ax1.set_title("Cropped Image")
        ax1.axis('off')
        
        # 显示缩放后的图像
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(self.resized_image, cmap='gray')
        ax2.set_title(f"Resized to {self.target_size}")
        ax2.axis('off')
        
        # 显示二值化后的图像
        ax3 = fig.add_subplot(gs[0, 2])
        if self.binary_image is not None:
            ax3.imshow(self.binary_image, cmap='gray')
            ax3.set_title("Binary Image")
        else:
            ax3.set_title("Binary Image (Not Available)")
        ax3.axis('off')
        
        # 显示归一化后的图像
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(self.normalized_image, cmap='gray')
        ax4.set_title("Normalized Image (Model Input)")
        ax4.axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"处理步骤可视化图像已保存至 {filename}")
    
    def get_normalized_image(self):
        """获取归一化后的图像，用于模型输入
        
        返回:
            归一化后的图像，适合模型输入
        """
        if self.normalized_image is None:
            print("尚未处理图像")
            return None
        
        # 直接返回处理好的图像，保持维度与detect.py期望的一致
        return torch.from_numpy(self.normalized_image).unsqueeze(0).unsqueeze(0)
    
    def get_output_dir(self):
        """获取当前输出目录
        
        返回:
            输出目录路径
        """
        return self.output_dir 