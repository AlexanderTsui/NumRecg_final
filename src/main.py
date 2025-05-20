import sys
import os
import time
import matplotlib.pyplot as plt
from module.mouse_capture import MouseCaptureModule
from module.image_process import ImageProcessModule
from module.detect import DigitRecognizer
from module.serial_communication import SerialCommunicationModule
from datetime import datetime

def create_output_dir():
    """创建以时间戳命名的输出目录"""
    # 确保基础输出目录存在
    if not os.path.exists("output"):
        os.makedirs("output")
    
    # 创建时间戳子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"创建输出目录: {output_dir}")
    
    return output_dir

def main():
    """手写数字识别系统主函数"""
    print("=== 手写数字识别系统 ===")
    print("正在初始化系统...")
    
    # 创建共享的输出目录
    output_dir = create_output_dir()
    
    # 创建鼠标笔迹捕捉模块
    capture = MouseCaptureModule(output_dir=output_dir)
    
    # 创建图像处理模块，共享同一个输出目录
    image_processor = ImageProcessModule(output_dir=output_dir)
    
    # 初始化数字识别器
    try:
        digit_recognizer = DigitRecognizer()
        digit_recognition_enabled = True
        print("数字识别模块初始化成功")
    except Exception as e:
        print(f"数字识别模块加载失败: {str(e)}")
        print("将继续运行但不进行数字识别")
        digit_recognition_enabled = False
    
    # 初始化串口通讯模块
    try:
        serial_module = SerialCommunicationModule()
        print("串口通讯模块初始化成功")
        # 交互式设置串口
        serial_enabled = serial_module.setup_interactive()
    except Exception as e:
        print(f"串口通讯模块初始化失败: {str(e)}")
        print("将继续运行但不进行串口通讯")
        serial_enabled = False
    
    try:
        # 启动笔迹捕捉
        capture.start()
        print("请使用鼠标左键开始书写，松开鼠标后有2秒时间可以继续书写")
        print("笔迹完成后系统会自动处理图像、保存图片，下次书写会自动清空画布")
        print(f"所有图像将保存到目录: {output_dir}")
        print("按Ctrl+C可以随时退出程序")
        
        # 记录上一次显示的状态
        last_complete_state = False
        
        # 主循环
        while True:
            # 检查是否完成笔迹
            current_complete_state = capture.is_stroke_complete()
            
            # 只有当状态从未完成变为完成时，才处理图像
            if current_complete_state and not last_complete_state:
                # 显示笔迹
                # capture.show_canvas()
                
                # 获取原始图像并处理
                original_image = capture.get_canvas()
                processed_image = image_processor.process_image(original_image)
                
                # 显示处理过程
                # image_processor.show_processing_steps()
                
                # 保存所有处理阶段的图像
                if capture.last_saved_file:
                    base_name = os.path.basename(capture.last_saved_file)
                    # 获取不带扩展名的文件名作为前缀
                    filename_prefix = os.path.splitext(base_name)[0]
                    saved_files = image_processor.save_all_processing_steps(filename_prefix)
                else:
                    # 如果原始图像未保存，则使用默认前缀保存处理图像
                    saved_files = image_processor.save_all_processing_steps()
                
                print("所有处理阶段的图像均已保存")
                
                # 进行数字识别
                if digit_recognition_enabled:
                    try:
                        # 获取归一化后的图像用于模型输入
                        normalized_image = image_processor.get_normalized_image()
                        
                        # 使用识别器预测数字
                        # 注意：这里与之前不同，因为图像处理模块现在返回的是torch.Tensor
                        digit, confidence = digit_recognizer.predict(normalized_image)
                        
                        print(f"识别结果: 数字 {digit}, 置信度: {confidence:.2f}")
                        
                        # 保存带有识别结果的图像
                        result_filename = os.path.join(output_dir, f"{filename_prefix}_result.png")
                        # 使用二值化后的图像（更清晰）
                        digit_recognizer.save_prediction_image(
                            image_processor.binary_image,
                            digit,
                            confidence,
                            result_filename
                        )
                        
                        # 通过串口发送识别结果
                        if serial_enabled:
                            serial_module.send_recognition_result(digit, confidence)
                            
                    except Exception as e:
                        print(f"数字识别失败: {str(e)}")
                        import traceback
                        traceback.print_exc()
            
            # 更新状态
            last_complete_state = current_complete_state
            
            # 短暂暂停，避免CPU占用过高
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 停止笔迹捕捉
        capture.stop()
        # 关闭串口连接
        if 'serial_module' in locals() and serial_enabled:
            serial_module.close()
        # 关闭所有图像窗口
        plt.close('all')
        print("程序已退出")
        print(f"所有图像已保存到: {output_dir}")

if __name__ == "__main__":
    main() 