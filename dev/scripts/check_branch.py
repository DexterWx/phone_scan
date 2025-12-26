import cv2
import numpy as np

# 8 邻域（顺时针，必须固定顺序）
NEIGHBORS = [(-1, 0), (-1, 1), (0, 1), (1, 1),
             (1, 0), (1, -1), (0, -1), (-1, -1)]


def calc_cn(img: np.ndarray, y: int, x: int) -> int:
    """
    计算像素 (y, x) 的连接数 CN
    """
    p = []
    for dy, dx in NEIGHBORS:
        p.append(1 if img[y + dy, x + dx] > 0 else 0)

    # 循环计算 |p(i) - p(i+1)|
    cn = 0
    for i in range(8):
        cn += abs(p[i] - p[(i + 1) % 8])

    return cn // 2


def count_endpoints_cn(img: np.ndarray) -> int:
    """
    使用 CN == 1 统计端点数
    """
    h, w = img.shape
    endpoints = 0

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if img[y, x] == 0:
                continue

            cn = calc_cn(img, y, x)
            if cn == 1:
                endpoints += 1

    return endpoints


def has_true_branch(img_path: str, block_size: int) -> bool:
    """
    判断是否存在一个 block_size x block_size 的删除区域，
    使得 CN 端点数增加 >= 3
    """
    assert block_size in (2, 3)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found")

    # 二值化（保险）
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    h, w = img.shape

    # 初始端点数（CN 定义）
    base_endpoints = count_endpoints_cn(img)

    # 提前取出前景像素坐标（减少循环）
    ys, xs = np.where(img > 0)
    fg_points = list(zip(ys, xs))

    half = block_size // 2

    for cy, cx in fg_points:
        y0 = cy - half
        x0 = cx - half
        y1 = y0 + block_size
        x1 = x0 + block_size

        # 越界跳过
        if y0 < 1 or x0 < 1 or y1 >= h - 1 or x1 >= w - 1:
            continue

        # 区域里没有前景，跳过
        if np.count_nonzero(img[y0:y1, x0:x1]) == 0:
            continue

        # 删除该区域
        tmp = img.copy()
        tmp[y0:y1, x0:x1] = 0

        new_endpoints = count_endpoints_cn(tmp)

        if new_endpoints - base_endpoints >= 3:
            return True

    return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python check_branch_cn.py <image_path> <block_size>")
        sys.exit(1)

    image_path = sys.argv[1]
    block_size = int(sys.argv[2])

    result = has_true_branch(image_path, block_size)
    print("detected" if result else "not detected")
