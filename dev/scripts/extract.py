import cv2
import numpy as np

def extract_red_via_lab(image_path, output_path):
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图片，请检查路径。")
        return

    # 2. 将 BGR 转换为 Lab 空间
    # L: 亮度, a: 红绿轴, b: 蓝黄轴
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)

    # 3. 对 a 通道进行对比度增强 (CLAHE)
    # 这一步是为了让变深的红色在视觉上更明显
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_a = clahe.apply(a)

    # 4. 阈值化处理（可选，用于生成掩膜）
    # 如果你想直接看“笔迹提取”后的样子，可以使用自适应阈值
    _, mask = cv2.threshold(enhanced_a, 140, 255, cv2.THRESH_BINARY)

    # 5. 为了保存成可见的 RGB 图片，我们需要做一些转换
    # 方案 A: 直接保存增强后的灰度 a 通道（越亮代表越红）
    # 方案 B: 将 a 通道映射回伪彩色，或者只保留红色像素
    
    # 这里我们采用方案 B：将非红色区域涂黑，保留原始红色
    result = cv2.bitwise_and(img, img, mask=mask)

    # 6. 保存结果
    # 我们保存两张图：一张是 a 通道灰度图，一张是提取后的彩色图
    cv2.imwrite("only_a_channel.jpg", enhanced_a)
    cv2.imwrite(output_path, result)
    
    print(f"处理完成！提取后的图片已保存为: {output_path}")
    print("同时也生成了 'only_a_channel.jpg'，亮度越高的地方代表红色特征越明显。")

# 使用方法
extract_red_via_lab('/Users/xu.wang/Desktop/20251223_194600_176.jpg', '/Users/xu.wang/Desktop/red.jpg')
