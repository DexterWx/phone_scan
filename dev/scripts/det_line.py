import cv2
import numpy as np
import os
import argparse
from PIL import Image
import pillow_heif

# 注册HEIF格式支持
pillow_heif.register_heif_opener()


def load_image(image_path):
    """支持多种格式的图片加载函数"""
    try:
        # 首先尝试用OpenCV读取（支持常见格式）
        img = cv2.imread(image_path)
        if img is not None:
            return img
        
        # 如果OpenCV失败，尝试用PIL读取（支持HEIC等格式）
        pil_image = Image.open(image_path)
        # 转换为RGB模式（PIL默认是RGB，OpenCV需要BGR）
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # 转换为numpy数组并调整颜色通道顺序（RGB -> BGR）
        img_array = np.array(pil_image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_bgr
        
    except Exception as e:
        raise FileNotFoundError(f"图片读取失败：{image_path}，错误：{str(e)}")


def find_outer_quad(
    img,
    block_size=51,
    c=10,
    morph_kernel=5,
    epsilon_factor=0.02,
    min_area_ratio=0.1
):
    """检测图片中最大的黑框四边形"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # background = cv2.GaussianBlur(gray, (151, 151), 0)
    # gray = cv2.divide(gray, background, scale=255)
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 自适应阈值
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c
    )
    # _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 形态学闭操作
    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 查找连通区域（外部轮廓）
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    best = None
    best_score = 0

    for ctn in contours:
        area = cv2.contourArea(ctn)
        if area < min_area_ratio * w * h:
            continue
        x, y, cw, ch = cv2.boundingRect(ctn)
        margin = min(x, y, w - x - cw, h - y - ch)
        score = area - margin * 50
        if score > best_score:
            best = ctn
            best_score = score

    if best is None:
        raise ValueError("未找到合适的外部黑框")

    # 多边形逼近
    epsilon = epsilon_factor * cv2.arcLength(best, True)
    approx = cv2.approxPolyDP(best, epsilon, True)

    if len(approx) != 4:
        hull = cv2.convexHull(best)
        approx = cv2.approxPolyDP(hull, epsilon, True)

    return gray, thresh, closed, approx


def draw_result(img, approx):
    """在原图上绘制检测到的四边形"""
    out = img.copy()
    cv2.drawContours(out, [approx], -1, (0, 0, 255), 4)
    for i, p in enumerate(approx):
        x, y = p[0]
        cv2.circle(out, (x, y), 10, (0, 255, 0), -1)
        cv2.putText(out, f"{i+1}", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return out

def preprocess_image(img: np.ndarray, resize_width: int = None, resize_height: int = None) -> np.ndarray:
    """图片预处理函数
    
    Args:
        img: OpenCV 图片数组
        resize_width: 目标宽度 (None 表示保持比例)
        resize_height: 目标高度 (None 表示保持比例)
    
    Returns:
        预处理后的图片数组
    """
    processed_img = img.copy()
    
    # 调整图片大小
    if resize_width is not None or resize_height is not None:
        if resize_width is not None and resize_height is not None:
            # 指定具体尺寸
            processed_img = cv2.resize(processed_img, (resize_width, resize_height))
        elif resize_width is not None:
            # 按宽度等比例缩放
            height, width = processed_img.shape[:2]
            new_height = int(height * resize_width / width)
            processed_img = cv2.resize(processed_img, (resize_width, new_height))
        elif resize_height is not None:
            # 按高度等比例缩放
            height, width = processed_img.shape[:2]
            new_width = int(width * resize_height / height)
            processed_img = cv2.resize(processed_img, (new_width, resize_height))
        
    
    return processed_img


def main():
    parser = argparse.ArgumentParser(description="检测图像中外部黑框的四个角点")
    parser.add_argument("image_path", type=str, help="输入图片路径")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="输出目录（默认：output）")
    parser.add_argument("--block_size", type=int, default=51,
                        help="自适应阈值 block size（奇数）")
    parser.add_argument("--c", type=int, default=10,
                        help="自适应阈值常数")
    parser.add_argument("--morph_kernel", type=int, default=5,
                        help="形态学闭操作核大小")
    parser.add_argument("--epsilon_factor", type=float, default=0.02,
                        help="多边形逼近精度因子")
    parser.add_argument("--min_area_ratio", type=float, default=0.1,
                        help="最小面积比例（相对整幅图）")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    img = load_image(args.image_path)
    img = preprocess_image(img, resize_width=2000)

    gray, thresh, closed, approx = find_outer_quad(
        img,
        block_size=args.block_size,
        c=args.c,
        morph_kernel=args.morph_kernel,
        epsilon_factor=args.epsilon_factor,
        min_area_ratio=args.min_area_ratio
    )

    result = draw_result(img, approx)

    # 输出中间结果
    cv2.imwrite(os.path.join(args.output_dir, "1_gray.jpg"), gray)
    cv2.imwrite(os.path.join(args.output_dir, "2_thresh.jpg"), thresh)
    cv2.imwrite(os.path.join(args.output_dir, "3_closed.jpg"), closed)
    cv2.imwrite(os.path.join(args.output_dir, "4_result.jpg"), result)

    print("✅ 检测完成！角点坐标如下：")
    for i, p in enumerate(approx):
        print(f"  点{i+1}: {p[0].tolist()}")
    print(f"结果图与中间过程已保存至: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
