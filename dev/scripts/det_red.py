import cv2
import numpy as np
import urllib.request
import os
from pathlib import Path

def load_image(image_path):
    """
    加载图片，支持本地路径和URL
    """
    if image_path.startswith('http'):
        # 从URL加载图片
        req = urllib.request.urlopen(image_path)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
    else:
        # 从本地路径加载图片
        img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"无法加载图片: {image_path}")
    
    return img

def extract_red_marks(image):
    """
    提取图像中的红色笔迹
    """
    # 转换到HSV色彩空间以便更好地检测红色
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 定义红色的HSV范围
    # 红色在HSV圆环上位于0度附近，需要定义两个范围来覆盖所有红色
    lower_red1 = np.array([0, 20, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 20, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # 创建红色掩码
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # 对掩码进行形态学操作以去除噪声并连接区域
    kernel = np.ones((3,3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    
    # 使用掩码提取红色区域
    red_result = cv2.bitwise_and(image, image, mask=red_mask)
    
    return red_result, red_mask

def main():
    # 获取用户输入
    import argparse
    parser = argparse.ArgumentParser(description='Extract red marks from image')
    parser.add_argument('image_path', help='Path or URL to the image')
    args = parser.parse_args()
    image_path = args.image_path
    
    try:
        # 加载图片
        print("正在加载图片...")
        image = load_image(image_path)
        
        # 提取红色笔迹
        print("正在提取红色笔迹...")
        red_result, red_mask = extract_red_marks(image)
        
        # 生成输出文件名
        if image_path.startswith('http'):
            output_filename = "red_extracted_from_url.jpg"
        else:
            path_obj = Path(image_path)
            output_filename = f"{path_obj.stem}_red_only{path_obj.suffix}"
        
        # 保存结果
        cv2.imwrite(output_filename, red_result)
        cv2.imwrite("red_mask_" + output_filename, red_mask)
        
        print(f"红色笔迹已保存为: {output_filename}")
        print(f"红色掩码已保存为: red_mask_{output_filename}")
        
    except Exception as e:
        print(f"处理过程中出错: {e}")

if __name__ == "__main__":
    main()