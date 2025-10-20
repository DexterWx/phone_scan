import sys
import json
import requests
import os
import shutil
from urllib.parse import urlparse
from PIL import Image
import io


class ImageProcessor:
    """图片处理模块"""
    
    @staticmethod
    def is_url(path):
        """判断路径是否为URL"""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    @staticmethod
    def convert_to_jpg(image_data, output_path):
        """将图片数据转换为JPG格式"""
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # 转换为RGB模式（JPG不支持透明度）
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            
            # 保存为JPG格式
            image.save(output_path, 'JPEG', quality=95)
            return True
        except Exception as e:
            print(f"图片转换失败: {e}")
            return False
    
    @staticmethod
    def download_image(url, output_path):
        """从URL下载图片并转换为JPG格式"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            image_data = response.content
            if ImageProcessor.convert_to_jpg(image_data, output_path):
                print(f"图片下载并转换成功: {output_path}")
                return True
            return False
        except Exception as e:
            print(f"图片下载失败: {e}")
            return False
    
    @staticmethod
    def process_local_image(src_path, dst_path):
        """读取本地图片并转换为JPG格式"""
        try:
            with open(src_path, 'rb') as f:
                image_data = f.read()
            
            if ImageProcessor.convert_to_jpg(image_data, dst_path):
                print(f"图片转换成功: {dst_path}")
                return True
            return False
        except Exception as e:
            print(f"图片转换失败: {e}")
            return False


class MarkDataProcessor:
    """标记数据处理模块"""
    
    @staticmethod
    def fetch_mark_data(mark_url):
        """获取标记数据"""
        try:
            print(f"正在获取标记数据: {mark_url}")
            response = requests.get(mark_url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"获取标记数据失败: {e}")
            return None
    
    @staticmethod
    def parse_mark_data(data):
        """解析标记数据"""
        try:
            page = data["body"]['scanJson']['pages'][0]
            boundary = page['objective_scan_area']
            rec_items = []
            assist_location = {
                "left": [],
                "right": []
            }
            for block in page['objective_blocks']:
                if 'assist_location_left_points' in block:
                    assist_location['left']+=block['assist_location_left_points']
                    assist_location['right']+=block['assist_location_right_points']
                for item in block['objective_items']:
                    rec_type = item['options_type']
                    if rec_type not in [1, 3]:
                        continue
                    rec_type = 1 if rec_type == 1 else 2
                    sub_options = item['options']
                    rec_items.append({
                        "rec_type": rec_type,
                        "sub_options": sub_options
                    })
            
            return {
                "boundary": boundary,
                "rec_items": rec_items,
                "assist_location": assist_location
            }
        except Exception as e:
            print(f"解析标记数据失败: {e}")
            return None


class FileManager:
    """文件管理模块"""
    
    @staticmethod
    def create_output_dir(output_dir):
        """创建输出目录"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"输出目录已创建: {output_dir}")
            return True
        except Exception as e:
            print(f"创建输出目录失败: {e}")
            return False
    
    @staticmethod
    def save_json(data, output_dir, filename="test.json"):
        """保存JSON数据到文件"""
        try:
            json_path = os.path.join(output_dir, filename)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"JSON数据保存成功: {json_path}")
            return True
        except Exception as e:
            print(f"JSON数据保存失败: {e}")
            return False
    
    @staticmethod
    def process_image(img_path, output_dir, filename="test.jpg"):
        """处理图片（URL或本地路径）"""
        img_output_path = os.path.join(output_dir, filename)
        
        if ImageProcessor.is_url(img_path):
            return ImageProcessor.download_image(img_path, img_output_path)
        else:
            return ImageProcessor.process_local_image(img_path, img_output_path)



if __name__ == "__main__":
    """主函数"""
    if len(sys.argv) != 4:
        print("用法: python get_mark.py <mark_url> <img_path> <output_dir>")
        sys.exit(1)
    
    mark_url = sys.argv[1]
    img_path = sys.argv[2]
    output_dir = sys.argv[3]
    
    # 创建输出目录
    if not FileManager.create_output_dir(output_dir):
        sys.exit(1)
    
    # 获取并解析标记数据
    data = MarkDataProcessor.fetch_mark_data(mark_url)
    
    mark = MarkDataProcessor.parse_mark_data(data)
    
    # 保存JSON数据
    FileManager.save_json(mark, output_dir)
    
    # 处理图片
    FileManager.process_image(img_path, output_dir)
    
    print("处理完成！")