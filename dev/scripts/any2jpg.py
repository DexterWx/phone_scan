import sys
import requests
import os
from urllib.parse import urlparse
from PIL import Image
import io

# 添加HEIC支持
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    print("HEIC支持已启用")
except ImportError:
    print("警告: 未安装pillow-heif，HEIC文件可能无法处理")


class ImageConverter:
    """图片转换模块"""
    
    @staticmethod
    def is_url(path):
        """判断路径是否为URL"""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    @staticmethod
    def get_exif_orientation(image):
        """获取EXIF方向信息"""
        try:
            if hasattr(image, '_getexif'):
                exif = image._getexif()
                if exif is not None:
                    orientation = exif.get(274)  # Orientation标签
                    return orientation
        except (AttributeError, KeyError, TypeError):
            pass
        return None
    
    @staticmethod
    def apply_exif_orientation(image):
        """根据EXIF信息旋转图片"""
        orientation = ImageConverter.get_exif_orientation(image)
        
        if orientation == 1:
            # 正常方向，不需要旋转
            return image
        elif orientation == 2:
            # 水平翻转
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # 旋转180度
            return image.rotate(180, expand=True)
        elif orientation == 4:
            # 垂直翻转
            return image.transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            # 水平翻转 + 逆时针旋转90度
            return image.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
        elif orientation == 6:
            # 顺时针旋转90度
            return image.rotate(-90, expand=True)
        elif orientation == 7:
            # 水平翻转 + 顺时针旋转90度
            return image.transpose(Image.FLIP_LEFT_RIGHT).rotate(-90, expand=True)
        elif orientation == 8:
            # 逆时针旋转90度
            return image.rotate(90, expand=True)
        else:
            # 未知方向或没有EXIF信息，返回原图
            return image
    
    @staticmethod
    def convert_to_jpg(image_data, output_path):
        """将图片数据转换为JPG格式"""
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # 应用EXIF方向信息
            image = ImageConverter.apply_exif_orientation(image)
            
            # 转换为RGB模式（JPG不支持透明度）
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            
            # 保存为JPG格式，不包含EXIF信息避免重复旋转
            image.save(output_path, 'JPEG', quality=95, optimize=True)
            return True
        except Exception as e:
            print(f"图片转换失败: {e}")
            return False
    
    @staticmethod
    def download_and_convert(url, output_path):
        """从URL下载图片并转换为JPG格式"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            image_data = response.content
            if ImageConverter.convert_to_jpg(image_data, output_path):
                print(f"图片下载并转换成功: {output_path}")
                return True
            return False
        except Exception as e:
            print(f"图片下载失败: {e}")
            return False
    
    @staticmethod
    def convert_local_image(src_path, dst_path):
        """读取本地图片并转换为JPG格式"""
        try:
            with open(src_path, 'rb') as f:
                image_data = f.read()
            
            if ImageConverter.convert_to_jpg(image_data, dst_path):
                print(f"图片转换成功: {dst_path}")
                return True
            return False
        except Exception as e:
            print(f"图片转换失败: {e}")
            return False


if __name__ == "__main__":
    """主函数"""
    if len(sys.argv) != 3:
        print("用法: python any2jpg.py <input_path_or_url> <output_path>")
        print("示例:")
        print("  python any2jpg.py input.png output.jpg")
        print("  python any2jpg.py input.heic output.jpg")
        print("  python any2jpg.py https://example.com/image.png output.jpg")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 处理图片转换
    if ImageConverter.is_url(input_path):
        # 如果是URL，下载并转换
        if not ImageConverter.download_and_convert(input_path, output_path):
            sys.exit(1)
    else:
        # 如果是本地路径，转换
        if not os.path.exists(input_path):
            print(f"输入文件不存在: {input_path}")
            sys.exit(1)
        
        if not ImageConverter.convert_local_image(input_path, output_path):
            sys.exit(1)
    
    print("转换完成！")