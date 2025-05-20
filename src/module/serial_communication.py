import serial
import serial.tools.list_ports
import time

class SerialCommunicationModule:
    """串口通讯模块，用于向外部设备发送识别结果"""
    
    def __init__(self):
        """初始化串口通讯模块"""
        self.serial_port = None
        self.is_connected = False
        self.port_name = None
        self.baud_rate = 9600  # 默认波特率
        
    def list_available_ports(self):
        """列出所有可用的串口"""
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            print("未检测到可用串口")
            return []
        
        print("检测到以下串口:")
        for i, port in enumerate(ports):
            print(f"{i+1}. {port.device} - {port.description}")
        
        return ports
    
    def open_port(self, port_name=None, baud_rate=9600):
        """打开指定的串口"""
        if self.is_connected:
            print(f"已经连接到串口 {self.port_name}")
            return True
            
        try:
            if port_name:
                self.port_name = port_name
            self.baud_rate = baud_rate
            
            self.serial_port = serial.Serial(
                port=self.port_name,
                baudrate=self.baud_rate,
                timeout=1
            )
            
            self.is_connected = True
            print(f"成功连接到串口 {self.port_name}，波特率：{self.baud_rate}")
            return True
        except Exception as e:
            print(f"连接串口 {port_name} 失败: {str(e)}")
            self.is_connected = False
            return False
    
    def close(self):
        """关闭串口连接"""
        if self.serial_port and self.is_connected:
            self.serial_port.close()
            self.is_connected = False
            print(f"已关闭串口 {self.port_name}")
    
    def send_data(self, data):
        """向串口发送数据"""
        if not self.is_connected or not self.serial_port:
            print("串口未连接，无法发送数据")
            return False
        
        try:
            # 确保数据是字节类型
            if isinstance(data, str):
                data = data.encode('utf-8')
            elif isinstance(data, int):
                data = str(data).encode('utf-8')
                
            self.serial_port.write(data)
            print(f"成功发送数据: {data}")
            return True
        except Exception as e:
            print(f"发送数据失败: {str(e)}")
            return False
    
    def send_recognition_result(self, digit, confidence):
        """发送识别结果"""
        if not self.is_connected:
            return False
            
        try:
            # 将识别结果格式化为易读格式
            message = f"DIGIT:{digit},CONF:{confidence:.2f}\r\n"
            result = self.send_data(message)
            return result
        except Exception as e:
            print(f"发送识别结果失败: {str(e)}")
            return False
    
    def setup_interactive(self):
        """交互式设置串口连接"""
        ports = self.list_available_ports()
        if not ports:
            print("没有检测到可用串口，将继续运行但不进行串口通讯")
            return False
        
        while True:
            try:
                selection = input("请输入要连接的串口编号 (输入'q'退出): ")
                if selection.lower() == 'q':
                    print("用户选择不使用串口功能")
                    return False
                
                port_index = int(selection) - 1
                if 0 <= port_index < len(ports):
                    selected_port = ports[port_index].device
                    baud_rate_input = input(f"请输入波特率 (默认 9600): ")
                    baud_rate = int(baud_rate_input) if baud_rate_input.strip() else 9600
                    
                    if self.open_port(selected_port, baud_rate):
                        return True
                else:
                    print("无效的选择，请重试")
            except ValueError:
                print("请输入有效的数字")
            except Exception as e:
                print(f"设置串口时出错: {str(e)}") 