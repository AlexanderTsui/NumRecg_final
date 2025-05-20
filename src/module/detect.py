import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageOps
import os
import numpy as np

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层
        x = self.conv2(x)  # 再来一次
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）
    
class DigitRecognizer:
    def __init__(self, model_path=None):
        """初始化手写数字识别器
        
        参数:
            model_path: 模型路径，如果为None则自动查找
        """
        if model_path is None:
            # 自动查找模型路径
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, 'models', 'model_Mnist.pth')
        
        # 初始化模型
        self.model = Net()
        
        # 尝试加载模型
        try:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()  # 设置为评估模式
            print(f"成功加载模型: {model_path}")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            raise
        
        # 定义图像转换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def preprocess_image_file(self, image_path):
        """从文件加载并预处理图像
        
        参数:
            image_path: 图像文件路径
            
        返回:
            预处理后的图像张量
        """
        # 加载图片并缩放到28x28
        image = Image.open(image_path).convert('L').resize((28, 28))
        # 反色
        image = ImageOps.invert(image)
        # 二值化
        image = image.point(lambda x: 0 if x < 128 else 255, '1')
        # 转为tensor和归一化
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0)  # 添加batch维度
    
    def preprocess_image_array(self, image_array):
        """从numpy数组预处理图像
        
        参数:
            image_array: numpy数组格式的图像，已经是28x28大小和黑底白字格式
            
        返回:
            预处理后的图像张量
        """
        # 确保是浮点型且值范围在0-1之间
        if image_array.dtype != np.float32:
            image_array = image_array.astype(np.float32)
        
        if image_array.max() > 1.0:
            image_array = image_array / 255.0
        
        # 转为tensor和归一化
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # 添加通道维度
        
        # 归一化
        image_tensor = transforms.Normalize((0.1307,), (0.3081,))(image_tensor)
        
        return image_tensor.unsqueeze(0)  # 添加batch维度
    
    def predict(self, image_tensor):
        """预测图像中的数字
        
        参数:
            image_tensor: 预处理后的图像张量
            
        返回:
            预测的数字和对应的概率
        """
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)
            return predicted.item(), confidence.item()
    
    def recognize_from_file(self, image_path):
        """从文件识别数字
        
        参数:
            image_path: 图像文件路径
            
        返回:
            预测的数字和对应的概率
        """
        image_tensor = self.preprocess_image_file(image_path)
        return self.predict(image_tensor)
    
    def recognize_from_array(self, image_array):
        """从numpy数组识别数字
        
        参数:
            image_array: numpy数组格式的图像
            
        返回:
            预测的数字和对应的概率
        """
        image_tensor = self.preprocess_image_array(image_array)
        return self.predict(image_tensor)
    
    def show_prediction(self, image, prediction, confidence, title=None):
        """显示图像和预测结果
        
        参数:
            image: 图像(PIL Image或numpy数组)
            prediction: 预测的数字
            confidence: 预测的置信度
            title: 自定义标题
        """
        plt.figure(figsize=(5, 5))
        
        if isinstance(image, np.ndarray):
            # 如果是numpy数组
            plt.imshow(image, cmap='gray')
        else:
            # 如果是PIL图像
            plt.imshow(image, cmap='gray')
        
        if title:
            plt.title(title)
        else:
            plt.title(f'识别结果: {prediction} (置信度: {confidence:.2f})')
            
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.1)
    
    def save_prediction_image(self, image, prediction, confidence, output_path):
        """保存带有预测结果的图像
        
        参数:
            image: 图像(PIL Image或numpy数组)
            prediction: 预测的数字
            confidence: 预测的置信度
            output_path: 输出文件路径
        """
        plt.figure(figsize=(5, 5))
        
        if isinstance(image, np.ndarray):
            # 如果是numpy数组
            plt.imshow(image, cmap='gray')
        else:
            # 如果是PIL图像
            plt.imshow(image, cmap='gray')
        
        plt.title(f'识别结果: {prediction} (置信度: {confidence:.2f})')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"预测结果图像已保存至: {output_path}")

def main():
    # 创建识别器
    recognizer = DigitRecognizer()
    
    # 获取用户输入
    image_path = input("请输入要识别的图片路径: ")
    
    try:
        # 识别图像
        prediction, confidence = recognizer.recognize_from_file(image_path)
        
        # 显示结果
        print(f"预测结果: {prediction}, 置信度: {confidence:.2f}")
        
        # 加载原图并显示结果
        image = Image.open(image_path).convert('L')
        recognizer.show_prediction(image, prediction, confidence)
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == '__main__':
    main() 