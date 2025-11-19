import os
import json
from pathlib import Path
import cv2
import numpy as np

def draw_mark_result(image, fill_items, rec_result, output_path):
    """
    在图像上绘制填涂结果
    
    Args:
        image: 原始图像
        fill_items: 填涂区域信息列表
        rec_result: 识别结果列表(True/False)
        output_path: 输出图像路径
    """
    # 创建图像副本
    result_img = image.copy()
    
    # 遍历每个填涂项
    for i, (is_filled, fill_item) in enumerate(zip(rec_result, fill_items)):
        coordinate = fill_item['coordinate']
        x, y, w, h = coordinate['x'], coordinate['y'], coordinate['w'], coordinate['h']
        
        # 根据是否填涂选择颜色和样式
        if is_filled:
            # 填涂了 - 实心红色
            color = (0, 0, 255)  # BGR格式，红色
            thickness = -1  # 实心填充
        else:
            # 未填涂 - 空心绿色
            color = (0, 255, 0)  # BGR格式，绿色
            thickness = 2  # 边框线宽
            
        # 绘制矩形
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, thickness)
        
        # 添加索引标签
        cv2.putText(result_img, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    # 保存结果图像
    cv2.imwrite(output_path, result_img)
    print(f"  结果图像已保存至: {output_path}")

def read_batch_test_data():
    """
    读取batch_test目录下的所有JSON文件
    """
    # 定义根目录路径
    base_path = Path("/Users/xu.wang/workspace/gitlab/phone_scan/dev/test_data/batch_test")
    
    # 检查目录是否存在
    if not base_path.exists():
        print(f"目录 {base_path} 不存在")
        return
    
    # 遍历所有子目录
    for subdir in sorted(base_path.iterdir()):
        if subdir.is_dir():
            print(f"处理目录: {subdir.name}")
            
            # 遍历目录中的所有JSON文件
            for json_file in subdir.glob("*.json"):
                print(f"  读取文件: {json_file.name}")
                
                # 读取JSON文件内容
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if data['result']['code'] != 0:
                            continue
                        
                        # 计算画布大小（基于所有坐标的最大范围）
                        max_x, max_y = 0, 0
                        for rec_result in data['result']['rec_results']:
                            for fill_item in rec_result.get('fill_items', []):
                                coord = fill_item['coordinate']
                                max_x = max(max_x, coord['x'] + coord['w'])
                                max_y = max(max_y, coord['y'] + coord['h'])
                        
                        # 创建一个新的空白图像
                        canvas = np.ones((max_y + 50, max_x + 50, 3), dtype=np.uint8) * 255
                        
                        # 在新图像上绘制所有识别结果
                        question_index = 1  # 题目序号从1开始
                        prev_y = -1  # 上一个题目的y坐标
                        y_threshold = 10  # 判断是否为同一组题目的y坐标阈值
                        
                        for idx, rec_result in enumerate(data['result']['rec_results']):
                            rec_type = rec_result.get('rec_tpye', 0)
                            fill_items = rec_result.get('fill_items', [])
                            result_flags = rec_result.get('rec_result', [])
                            
                            # 获取当前题目第一个选项的y坐标，用于判断是否是新题目
                            if fill_items:
                                current_y = fill_items[0]['coordinate']['y']
                                # 如果与上一个题目的y坐标差超过阈值，则认为是新题目
                                if abs(current_y - prev_y) > y_threshold:
                                    prev_y = current_y
                                else:
                                    # 同一题目不需要增加序号
                                    pass
                            
                            # 绘制当前识别结果的所有填涂项
                            for i, (is_filled, fill_item) in enumerate(zip(result_flags, fill_items)):
                                coordinate = fill_item['coordinate']
                                x, y, w, h = coordinate['x'], coordinate['y'], coordinate['w'], coordinate['h']
                                
                                # 根据是否填涂选择颜色和样式
                                if is_filled:
                                    # 填涂了 - 实心红色
                                    color = (0, 0, 255)  # BGR格式，红色
                                    thickness = -1  # 实心填充
                                else:
                                    # 未填涂 - 空心绿色
                                    color = (0, 255, 0)  # BGR格式，绿色
                                    thickness = 2  # 边框线宽
                                    
                                # 绘制矩形
                                cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
                                
                                # 只在每组题的第一个选项左侧添加序号
                                if i == 0:
                                    # 找到该组题目的最小x和y坐标
                                    min_x = min([item['coordinate']['x'] for item in fill_items])
                                    min_y = min([item['coordinate']['y'] for item in fill_items])
                                    max_y_items = max([item['coordinate']['y'] + item['coordinate']['h'] for item in fill_items])
                                    
                                    # 在题目最左侧稍偏右一点的位置添加序号
                                    text_x = min_x - 20 if min_x > 20 else 5
                                    text_y = min_y + (max_y_items - min_y) // 2
                                    
                                    cv2.putText(canvas, str(question_index), (text_x, text_y), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                                    
                                    # 更新题目序号
                                    question_index += 1
                        
                        # 生成输出文件名（与JSON文件同级目录）
                        output_filename = f"{json_file.stem}_visualization.jpg"
                        output_path = json_file.parent / output_filename
                        
                        # 保存结果图像
                        cv2.imwrite(str(output_path), canvas)
                        print(f"  可视化结果已保存至: {output_path}")
                        
                except json.JSONDecodeError as e:
                    print(f"    错误: 无法解析JSON文件 {json_file.name}: {e}")
                except Exception as e:
                    print(f"    错误: 读取文件 {json_file.name} 失败: {e}")

if __name__ == "__main__":
    read_batch_test_data()