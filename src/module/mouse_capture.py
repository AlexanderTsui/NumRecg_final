import numpy as np
import time
import threading
from pynput import mouse
import cv2
import matplotlib.pyplot as plt
import os
from datetime import datetime

class MouseCaptureModule:
    def __init__(self, canvas_size=(2560, 1440), output_dir=None):
        """初始化鼠标笔迹捕捉模块
        
        参数:
            canvas_size: 画布大小，默认为2560x1440像素
            output_dir: 输出目录，如果为None则自动创建时间戳目录
        """
        # 画布大小
        self.canvas_size = canvas_size
        # 创建空白画布
        self.canvas = np.zeros((self.canvas_size[1], self.canvas_size[0]), dtype=np.uint8)
        # 存储笔迹轨迹点
        self.points = []
        # 是否正在绘制
        self.is_drawing = False
        # 计时器
        self.timer = None
        # 笔迹是否完成
        self.is_complete = False
        # 鼠标监听器
        self.listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click
        )
        # 笔迹线条粗细
        self.line_thickness = 20
        
        # 设置输出目录
        if output_dir is None:
            self.output_dir = self._create_timestamp_dir()
        else:
            self.output_dir = output_dir
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                
        # 是否已经处理了当前完成的笔迹
        self.is_processed = False
        # 最近保存的文件路径
        self.last_saved_file = None
        # 当前笔迹是否已显示
        self.is_displayed = False
    
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
    
    def start(self):
        """开始监听鼠标事件"""
        self.listener.start()
        print("鼠标笔迹捕捉已启动，请使用鼠标左键在屏幕上书写数字")
    
    def stop(self):
        """停止监听鼠标事件"""
        if self.listener.is_alive():
            self.listener.stop()
            self.listener.join()
        if self.timer:
            self.timer.cancel()
        print("鼠标笔迹捕捉已停止")
    
    def _on_move(self, x, y):
        """鼠标移动事件处理函数"""
        if self.is_drawing:
            # 自动重置上一次完成的画布
            if self.is_complete and not self.is_processed:
                self.reset()
                self.is_processed = True
                
            # 将笔迹点添加到轨迹中
            self.points.append((x, y))
            # 在画布上绘制笔迹
            if len(self.points) > 1:
                pt1 = self.points[-2]
                pt2 = self.points[-1]
                # 确保点在画布范围内
                if (0 <= pt1[0] < self.canvas_size[0] and 
                    0 <= pt1[1] < self.canvas_size[1] and
                    0 <= pt2[0] < self.canvas_size[0] and 
                    0 <= pt2[1] < self.canvas_size[1]):
                    # 注意：cv2的坐标是(y, x)，所以要交换
                    cv2.line(self.canvas, pt1, pt2, 255, self.line_thickness)
    
    def _on_click(self, x, y, button, pressed):
        """鼠标点击事件处理函数"""
        if button == mouse.Button.left:
            if pressed:
                # 如果上一次笔迹已完成但未被处理，重置画布
                if self.is_complete:
                    self.reset()
                
                # 左键按下开始绘制
                self.is_drawing = True
                self.is_complete = False
                self.is_processed = False
                self.is_displayed = False
                self.points.append((x, y))
                
                # 如果存在计时器，取消它
                if self.timer:
                    self.timer.cancel()
                    self.timer = None
            else:
                # 左键释放暂停绘制
                self.is_drawing = False
                # 提示用户可以在两秒内继续书写
                print("您已松开鼠标，可以在2秒内继续书写，超过2秒将完成本次笔迹记录")
                # 启动2秒计时器
                self.timer = threading.Timer(2.0, self._on_timer_complete)
                self.timer.start()
    
    def _on_timer_complete(self):
        """计时器完成事件，表示笔迹已完成"""
        self.is_complete = True
        self._process_completed_stroke()
        # 保存笔迹图片
        self.last_saved_file = self.save_canvas()
        print("笔迹记录完成！已保存笔迹图片，下次书写将自动清空画布")
    
    def _process_completed_stroke(self):
        """处理完成的笔迹"""
        # 对笔迹进行加粗处理
        kernel = np.ones((3, 3), np.uint8)
        self.canvas = cv2.dilate(self.canvas, kernel, iterations=1)
    
    def get_canvas(self):
        """获取当前画布"""
        return self.canvas.copy()
    
    def reset(self):
        """重置画布和笔迹状态"""
        self.canvas = np.zeros((self.canvas_size[1], self.canvas_size[0]), dtype=np.uint8)
        self.points = []
        self.is_drawing = False
        if self.timer:
            self.timer.cancel()
            self.timer = None
        self.is_displayed = False
        print("画布已重置，可以开始新的书写")
    
    def show_canvas(self):
        """显示当前画布"""
        if not self.is_displayed:
            # 关闭之前的图像窗口
            plt.close('all')
            # 使用matplotlib正确显示图像
            plt.figure(figsize=(10, 10))
            # 转置画布以保持正确的长宽比
            plt.imshow(self.canvas, cmap='gray')
            plt.title("Written Number")
            plt.axis('off')
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
            self.is_displayed = True
    
    def save_canvas(self):
        """保存当前画布为图片"""
        index = 1
        while True:
            # 使用序号生成文件名，避免重复
            filename = os.path.join(self.output_dir, f"handwritten_{index:03d}.png")
            if not os.path.exists(filename):
                break
            index += 1
            
        # 保存原始大小图片
        cv2.imwrite(filename, self.canvas)
        print(f"笔迹图片已保存至 {filename}")
        return filename
    
    def is_stroke_complete(self):
        """检查笔迹是否完成"""
        return self.is_complete
    
    def get_last_saved_file(self):
        """获取最近保存的文件路径"""
        return self.last_saved_file
    
    def get_output_dir(self):
        """获取当前输出目录
        
        返回:
            输出目录路径
        """
        return self.output_dir 