#!/usr/bin/env python3
"""
检测图片中的四边形
流程: 灰度 -> 高斯模糊 -> 自适应二值化 -> 形态学处理 -> 轮廓检测 -> 多边形近似
"""

import cv2
import numpy as np
import sys
import os


def detect_rectangles(image_path: str):
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return

    original = img.copy()
    h, w = img.shape[:2]

    # 1. 灰度转换
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. 自适应二值化
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )

    # 4. 形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. 找外轮廓
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. 筛选四边形（允许4-8个顶点，用凸包拟合成四边形）
    quads = []
    for contour in contours:
        # 多边形近似
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # 允许4-8个顶点
        if 4 <= len(approx) <= 12:
            # 计算凸包
            hull = cv2.convexHull(contour)
            # 对凸包再做多边形近似，强制拟合成4个点
            hull_peri = cv2.arcLength(hull, True)
            hull_approx = cv2.approxPolyDP(hull, 0.02 * hull_peri, True)

            # 如果凸包近似后还不是4个点，用最小外接矩形
            if len(hull_approx) == 4:
                quad = hull_approx
            else:
                rect = cv2.minAreaRect(hull)
                quad = np.int32(cv2.boxPoints(rect)).reshape(-1, 1, 2)

            area = cv2.contourArea(quad)
            if area > 400:  # 过滤太小的
                quads.append({
                    'contour': quad,
                    'area': area,
                    'original_vertices': len(approx)
                })

    # 7. 按面积排序，取前5个
    quads.sort(key=lambda x: x['area'], reverse=True)
    quads = quads[:5]

    print(f"检测到 {len(quads)} 个四边形")

    # 8. 绘制结果
    result = original.copy()
    for i, q in enumerate(quads):
        corners = q['contour'].reshape(-1, 2)
        # 绘制四边形
        cv2.polylines(result, [q['contour']], True, (0, 255, 0), 2)
        # 绘制角点
        for corner in corners:
            cv2.circle(result, tuple(corner), 5, (0, 0, 255), -1)
        # 标注序号和面积
        center = corners.mean(axis=0).astype(int)
        cv2.putText(result, f"#{i+1}", (center[0] - 20, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 9. 创建可视化面板
    gray_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    blurred_vis = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    binary_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    morphed_vis = cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR)

    cv2.putText(gray_vis, "Gray", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(blurred_vis, "Gaussian", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(binary_vis, "Binary", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(morphed_vis, "Morphology", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(result, "Result", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    row1 = np.hstack([gray_vis, blurred_vis, binary_vis])
    row2 = np.hstack([morphed_vis, result, original])
    combined = np.vstack([row1, row2])

    # 10. 保存结果
    input_dir = os.path.dirname(image_path)
    input_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(input_dir, f"{input_name}_result.jpg")
    cv2.imwrite(output_path, combined)
    print(f"结果已保存到: {output_path}")

    return quads


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "/Users/xu.wang/Desktop/wecom-temp-16ea18f7bb842d4639fc9f42c066423b.jpg"

    detect_rectangles(image_path)
