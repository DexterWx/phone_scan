#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import pandas as pd
import argparse
from typing import List, Dict, Tuple

def load_recognition_results(result_dir: str) -> Dict[int, List[Dict]]:
    """
    加载识别结果JSON文件
    """
    results = {}
    
    # 遍历1-20文件夹
    for i in range(1, 21):
        folder_path = os.path.join(result_dir, str(i))
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹 {folder_path} 不存在")
            continue
            
        folder_results = []
        
        # 遍历文件夹内的JSON文件
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                json_path = os.path.join(folder_path, filename)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        folder_results.append({
                            'filename': filename,
                            'image_path': data.get('image_path', ''),
                            'rec_results': data.get('result', {}).get('rec_results', [])
                        })
                except Exception as e:
                    print(f"读取文件 {json_path} 出错: {e}")
        
        results[i] = folder_results
    
    return results

def load_label_results(label_dir: str) -> Dict[int, pd.DataFrame]:
    """
    加载标注结果Excel文件
    """
    labels = {}
    
    for i in range(1, 21):
        excel_path = os.path.join(label_dir, f"{i}.xlsx")
        if not os.path.exists(excel_path):
            print(f"警告: 文件 {excel_path} 不存在")
            continue
            
        try:
            df = pd.read_excel(excel_path)
            labels[i] = df
        except Exception as e:
            print(f"读取文件 {excel_path} 出错: {e}")
    
    return labels

def calculate_question_type_accuracy(rec_results: List, label_answers: pd.Series) -> Tuple[float, List[str]]:
    """
    计算每道题的正确率
    题目类型分布：
    - 1-10: abcd单选 (rec_result[0:4])
    - 11-20: abcdefghi单选 (rec_result[0:9]) 
    - 21-30: abcd多选 (rec_result[0:4])
    - 31-40: abcdefghi多选 (rec_result[0:9])
    - 41-50: tf单选 (rec_result[0:2])
    """
    total_questions = len(label_answers)
    correct_count = 0
    wrong_questions = []
    
    for i, label_answer in enumerate(label_answers):
        question_num = i + 1
        
        # 判断题目类型和选项范围
        if 1 <= question_num <= 10:  # abcd单选
            option_range = 4
            is_multiple = False
        elif 11 <= question_num <= 20:  # abcdefghi单选
            option_range = 9
            is_multiple = False
        elif 21 <= question_num <= 30:  # abcd多选
            option_range = 4
            is_multiple = True
        elif 31 <= question_num <= 40:  # abcdefghi多选
            option_range = 9
            is_multiple = True
        elif 41 <= question_num <= 50:  # tf单选
            option_range = 2
            is_multiple = False
        else:
            continue
        
        # 获取对应的识别结果
        if question_num - 1 < len(rec_results):
            rec_result = rec_results[question_num - 1]['rec_result'][:option_range]
        else:
            rec_result = [False] * option_range
        
        # 解析标注答案
        if pd.isna(label_answer) or label_answer == '' or str(label_answer).lower().strip() == 'nan':
            # 空答案表示所有选项都应该是false
            expected_result = [False] * option_range
        else:
            expected_result = [False] * option_range
            answer_str = str(label_answer).lower().strip()
            
            if is_multiple:
                # 多选：解析多个选项
                for char in answer_str:
                    if char.isalpha():
                        char_idx = ord(char) - ord('a')
                        if 0 <= char_idx < option_range:
                            expected_result[char_idx] = True
            else:
                # 单选：只有一个选项为true
                if 41 <= question_num <= 50 and answer_str in ['t', 'f']:
                    # 只有41-50题是判断题：t对应True(索引0)，f对应False(索引1)
                    if answer_str == 't':
                        expected_result[0] = True  # True选项
                    elif answer_str == 'f':
                        expected_result[1] = True  # False选项
                elif answer_str.isalpha() and len(answer_str) == 1:
                    # 1-40题中的字母是选项标识
                    char_idx = ord(answer_str) - ord('a')
                    if 0 <= char_idx < option_range:
                        expected_result[char_idx] = True
        
        # 比较识别结果和期望结果
        if rec_result == expected_result:
            correct_count += 1
        else:
            wrong_questions.append(f"第{question_num}题: 期望{expected_result}, 识别{rec_result}")
    
    accuracy = correct_count / total_questions if total_questions > 0 else 0.0
    return accuracy, wrong_questions

def main():
    parser = argparse.ArgumentParser(description='统计试卷识别正确率')
    parser.add_argument('result_dir', help='识别结果文件夹路径')
    parser.add_argument('label_dir', help='标注结果文件夹路径')
    
    args = parser.parse_args()
    
    print("正在加载识别结果...")
    rec_results = load_recognition_results(args.result_dir)
    
    print("正在加载标注结果...")
    label_results = load_label_results(args.label_dir)
    
    print("\n开始统计正确率...")
    
    total_accuracy_sum = 0.0
    total_exams = 0
    all_wrong_questions = []
    
    for exam_num in range(1, 21):
        if exam_num not in rec_results or exam_num not in label_results:
            print(f"试卷{exam_num}: 数据不完整，跳过")
            continue
        
        exam_rec_results = rec_results[exam_num]
        exam_labels = label_results[exam_num]
        
        if not exam_rec_results:
            print(f"试卷{exam_num}: 无识别结果，跳过")
            continue
        
        # 计算每张图片的正确率
        exam_accuracies = []
        exam_images = []
        
        for img_data in exam_rec_results:
            image_path = img_data['image_path']
            img_rec_results = img_data['rec_results']
            
            accuracy, wrong_questions = calculate_question_type_accuracy(img_rec_results, exam_labels['答案'])
            exam_accuracies.append(accuracy)
            exam_images.append(image_path)
            all_wrong_questions.extend([f"试卷{exam_num} - {image_path}: {q}" for q in wrong_questions])
        
        # 计算试卷平均正确率
        if exam_accuracies:
            exam_avg_accuracy = sum(exam_accuracies) / len(exam_accuracies)
            total_accuracy_sum += exam_avg_accuracy
            total_exams += 1
    
    # 输出明细和总准确率
    if all_wrong_questions:
        print("错误题目详情:")
        
        # 按图片分组显示错误
        current_image = None
        for wrong_q in reversed(all_wrong_questions):
            # 提取图片路径
            image_path = wrong_q.split(": ")[0].split(" - ", 1)[1]
            question_detail = wrong_q.split(": ", 1)[1]
            
            if current_image != image_path:
                current_image = image_path
                print(f"\n{image_path}:")
            
            print(f"  {question_detail}")
    
    if total_exams > 0:
        overall_accuracy = total_accuracy_sum / total_exams
        print(f"\n总准确率: {overall_accuracy:.2%}")
    else:
        print("没有可统计的数据")

if __name__ == "__main__":
    main()
