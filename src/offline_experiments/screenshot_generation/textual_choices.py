# -*- coding: utf-8 -*-
# Copyright (c) 2024 OSU Natural Language Processing Group
# 版权声明和许可证信息...

import json
import os
import jsonlines
import base64
import numpy as np
import cv2
import copy
from tqdm import tqdm
import argparse
import supervision as sv
import torch
import pickle as pkl

# 导入自定义工具函数
import sys
sys.path.append("/Users/bytedance/workspace/Mind2Web/SeeAct")
from src.data_utils.image_utils import convert_elements2detections  # 将元素转换为检测框
from src.data_utils.image_utils import extract_topk_elements, extract_elements_by_ids  # 提取元素相关函数
from src.data_utils.image_utils import batch_elements_by_locality, batch_elements_by_locality_16_16_17  # 基于位置关系批处理元素
from src.data_utils.format_prompt_utils import data_format_input_multichoice  # 格式化多选输入数据

def run(args):
    # 加载选定的任务ID集合
    with open(args.selected_set_task_id_path, 'rb') as f:
        selected_set_task_id_dict = pkl.load(f)

    selected_task_ids = selected_set_task_id_dict[args.split]

    # 设置截图源数据路径
    screenshot_dump_path = args.screenshot_dump_path

    # 创建输出目录
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 加载查询源数据
    query_source_path = args.query_source_path
    with open(query_source_path, 'r') as f:
        all_queries = json.load(f)

    # 遍历每个任务并生成截图
    for i, task in tqdm(enumerate(all_queries)):
        if len(task) == 2:
            continue
        
        # 获取任务ID和动作ID
        task_action_id = task[0]
        task_id, action_id = task_action_id.strip().split("_")
        if task_id not in selected_task_ids:
            continue

        # 加载单个截图数据
        single_screenshot_path = os.path.join(screenshot_dump_path, task_id, "processed/screenshot.json")
        if os.path.exists(single_screenshot_path):
            with open(single_screenshot_path) as f:
                scrshots_task = json.load(f)
        else:
            print("No Folder: ", single_screenshot_path)
            continue

        # 创建任务输出目录
        task_dir = os.path.join(output_dir, task_action_id)
        if not os.path.exists(task_dir):
            os.mkdir(task_dir)

        image_dir = os.path.join(output_dir, task_action_id, "images")
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

        # 整理截图数据
        actid2scrshots_task = {}
        for scrshot in scrshots_task:
            tsd_act_uid = scrshot["action_uid"]
            actid2scrshots_task[tsd_act_uid] = scrshot
        scrshot = actid2scrshots_task[action_id]

        inference_batches = task[1]
        sample = task[2]

        # 处理图像数据
        bef_tsd = scrshot["before"]["screenshot"]
        bef_tsd = np.frombuffer(base64.b64decode(bef_tsd), np.uint8)
        bef_img = cv2.imdecode(bef_tsd, cv2.IMREAD_COLOR)

        # 收集所有元素（正例和负例）
        all_elements = []
        positive_elements = sample['pos_candidates']
        negative_elements = sample['neg_candidates']
        all_elements.extend(positive_elements)
        all_elements.extend(negative_elements)

        # 提取前50个元素并按位置批处理
        top_50_elements = extract_topk_elements(all_elements, k=50)
        if args.num_choice == -1:
            choice_batches = batch_elements_by_locality_16_16_17(top_50_elements)
        else:
            choice_batches = batch_elements_by_locality(top_50_elements, num_choices=args.num_choice)

        # 处理每个批次
        to_run = []
        for batch_idx, candidate_elements in enumerate(choice_batches):
            temp = copy.deepcopy(sample)

            # 准备问题和选项数据
            candidate_element_ids = [item['backend_node_id'] for item in candidate_elements]
            seq_context, seq_in, _, choices, node_to_keep = data_format_input_multichoice(
                temp, candidate_element_ids, -1, keep_html_brackets=True
            )
            
            # 更新临时数据
            temp['context_html'] = seq_context
            temp['context_node_ids'] = copy.deepcopy(list(node_to_keep))
            temp['question'] = seq_in
            temp['choices'] = choices
            temp['image_path'] = os.path.join("", task_action_id, "images")

            # 重新排序选项和元素
            candidate_element_ids = [item[0] for item in choices]
            candidate_elements = extract_elements_by_ids(all_elements, ids=candidate_element_ids)

            # 生成标注图像
            candidate_detections = convert_elements2detections(candidate_elements)
            candidate_labels = [chr(i+65) for i in range(len(candidate_detections))]

            # 处理和保存图像
            annotated_image = bef_img.copy()
            annotated_image = sv.crop_image(image=annotated_image, xyxy=np.array(
                [
                    0,
                    max(0, min(candidate_detections.xyxy[:, 1])-1024),
                    annotated_image.shape[1],
                    min(annotated_image.shape[0], max(candidate_detections.xyxy[:, 3])+1024)
                ]
            ))
            bef_fn = os.path.join(image_dir, "{}.jpg".format(batch_idx))
            try:
                cv2.imwrite(bef_fn, annotated_image)
            except:
                continue
            to_run.append(temp)
            
        # 保存处理结果
        pred_path = os.path.join(task_dir, "queries.jsonl")
        with jsonlines.open(pred_path, mode='w') as writer:
            writer.write_all(to_run)

# 主函数入口
if __name__ == '__main__':
    # 设置命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_choice', type=int, default=-1)  # 选项数量，-1表示使用16/16/17的分配方式
    parser.add_argument('--split', type=str, default="test_website")  # 数据集分割
    parser.add_argument('--selected_set_task_id_path', type=str, default="../data/seeact_source_data/30_selected.pkl")  # 选定任务ID路径
    parser.add_argument('--screenshot_dump_path', type=str, default="../data/screenshot_source/")  # 截图源数据路径
    parser.add_argument('--output_dir', type=str, default="../data/30_selected_tasks/exp4_whole")  # 输出目录
    parser.add_argument('--query_source_path', type=str,
                        default="../data/seeact_source_data/test_website_outputs_top50.json")  # 查询源数据路径
    my_args = parser.parse_args()
    run(my_args)