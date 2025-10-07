import os
import json
import re
import time
import random
from typing import List, Dict, Any, Tuple
import dashscope
from http import HTTPStatus
from prompt_AD_test import prompt_AD_tester, resize_image, mask_to_bbox, calculate_iou, encode_image
import cv2
import numpy as np
from earlystopping import EarlyStopping
import sys
from contextlib import contextmanager

class PromptOptimizer:
    def __init__(self, base_prompt, api_key, patience=5, min_improvement=0, save_path=""):
        self.base_prompt = base_prompt
        self.current_prompt = base_prompt
        self.optimization_history = []
        dashscope.api_key = api_key
        self.early_stopping = EarlyStopping(patience, min_improvement, base_prompt, save_path)
        
    def classify_errors(self, error_results: List[Dict]) -> Dict[str, List[Dict]]:
        """将错误结果分类为不同的错误类型"""
        error_categories = {
            "false_positive": [],  # 正常样本误判为异常
            "false_negative": [],  # 异常样本漏检
            "bbox_mislocalization": [],  # 边界框定位错误
            "format_violation": [],  # 输出格式不符合要求
            "type_misclassification": [],  # 异常类型误分类
        }
        
        for result in error_results:
            true_label = result['true_label']
            pred_label = result['predicted_label']
            raw_response = result['raw_response']
            
            if true_label == "normaly" and pred_label == "normal":
                continue

            # 检查格式是否符合要求
            if not re.search(r'<answer>.*?</answer>', raw_response, re.DOTALL):
                error_categories['format_violation'].append(result)
            # 正常样本误判为异常
            elif true_label == "normal" and pred_label == "anomaly":
                error_categories['false_positive'].append(result)
            # 异常样本漏检
            elif true_label == "anomaly" and pred_label == "normal":
                error_categories['false_negative'].append(result)
            elif result['predicted_bboxes']:
                # 异常类型误分类
                if result['predicted_bboxes'][0]['label'] != result.get('defect_type'):
                    error_categories['type_misclassification'].append(result)
                else:
                    # 边界框标注异常
                    mask = resize_image(os.path.join(DATASET_PROCESSED_DIR, result['class'], \
                                        result['defect_type'], 'mask', result['image_name']))
                    iou = calculate_iou(result['predicted_bboxes'], mask_to_bbox(mask))
                    if iou < 0.5:
                        result['iou'] = iou
                        error_categories['bbox_mislocalization'].append(result)
            

        return error_categories
    
    def select_typical_errors(self, error_categories: Dict, max_per_category: int = 3) -> List[Dict]:
        """从每个错误类别中选择典型样本（优先选严重或置信度极端的）"""
        typical_errors = []
        for category, errors in error_categories.items():
            if errors:
                if category == "bbox_mislocalization":
                    errors = sorted(errors, key=lambda x: x.get("iou", 0))  # IoU越低越严重
                elif category == "false_positive":
                    errors = sorted(errors, key=lambda x: x.get("confidence", 0), reverse=True)  # 置信度高但误判
                elif category == "false_negative":
                    errors = sorted(errors, key=lambda x: x.get("confidence", 0))  # 置信度低但漏检
                elif category == "type_misclassification":
                    continue # 这里先不处理分类错误，重点关注是否有故障
                    errors = sorted(errors, key=lambda x: x.get("confidence", 0))  # 置信度低但误分类
                elif category == "format_violation":
                    random.shuffle(errors) # 打乱随机采样
                selected = errors[:max_per_category]

                # 为每一个错误类型添加参考图片
                for error in selected:
                    error['category'] = category
                    if category != 'format_violation':# 在图片上绘制标注框
                        self._draw_detection_boxes(error)
                    self._prepare_visual_info(error)

                # 格式违规只选一个典型案例
                if category == 'format_violation':
                    selected = [selected[0]]
                typical_errors.extend(selected)
                print(f"类别 {category}: 选择了 {len(selected)} 个典型错误")
        return typical_errors
    
    def _prepare_visual_info(self, error: Dict):
        """为每张图片提供图片描述"""
        
        pred_bbox = error.get('pred_bbox', [])
        gt_label = error.get('defect_type', '')
        pred_label = error.get('predicted bboxes', '')[0]['label'] if error.get('predicted bboxes') else ''
        confidence = error.get('confidence', 0)
        iou = error.get('iou', 0)
        category = error.get('category', '')
        
        # 建立详细基本描述
        visual_description = f"Error Type: {category}\n"
        # 边框定位错误
        if category == "bbox_mislocalization":
            visual_description += f"【Localization Error】\n"
            visual_description += f"IoU: {iou:.1f} (threshold typically 0.5)\n"
            if pred_bbox and len(pred_bbox) == 4:
                x1, y1, x2, y2 = map(int, pred_bbox)
                w, h = x2-x1, y2-y1
                center_x, center_y = (x1+x2)//2, (y1+y2)//2
                visual_description += f"Predicted BBox: position ({x1}, {y1}) to ({x2}, {y2}), size {w}×{h}, center ({center_x}, {center_y})\n"
                visual_description += f"Predicted Label: {pred_label}\n"
            visual_description += "Visualization: Real anomaly areas are marked with blue transparent mask and yellow borders, predicted bbox is shown with red border, predicted anomaly category is labeled\n"
            visual_description += "Problem Analysis: The model detected the anomaly but localization is inaccurate, with insufficient overlap between predicted and ground truth boxes\n"
        # 正常样本误判为异常
        elif category == "false_positive":
            visual_description += f"【False Positive Error】\n"
            visual_description += f"Model Confidence: {confidence:.1f}\n"
            if pred_bbox and len(pred_bbox) == 4:
                x1, y1, x2, y2 = map(int, pred_bbox)
                w, h = x2-x1, y2-y1
                center_x, center_y = (x1+x2)//2, (y1+y2)//2
                visual_description += f"Predicted BBox: position ({x1}, {y1}) to ({x2}, {y2}), size {w}×{h}, center ({center_x}, {center_y})\n"
                visual_description += f"Predicted Label: {pred_label}\n"
            visual_description += "Visualization: Only red border shows the predicted bbox and its label in a normal image\n"
            visual_description += "Ground Truth: This area is actually normal or does not contain the predicted anomaly\n"
            visual_description += "Problem Analysis: The model incorrectly detected an anomaly in a normal area\n"
        # 异常样本漏检
        elif category == "false_negative":
            visual_description += f"【False Negative Error】\n"
            visual_description += f"Model Confidence: {confidence:.1f}\n"
            visual_description += "Visualization: Real anomaly areas are marked with blue transparent mask and yellow borders.\n"
            visual_description += "Problem Analysis: The model missed a real anomaly that exists in the image\n"
        # 异常类型误分类
        elif category == "type_misclassification":
            visual_description += f"【Classification Error】\n"
            visual_description += f"Model Confidence: {confidence:.1f}\n"
            if pred_bbox and len(pred_bbox) == 4:
                x1, y1, x2, y2 = map(int, pred_bbox)
                visual_description += f"Predicted BBox: position ({x1}, {y1}) to ({x2}, {y2})\n"
                visual_description += f"Predicted Label: {pred_label}\n"
            visual_description += f"Problem Analysis: The model misclassified {gt_label} as {pred_label}\n"
        # 输出格式不符合要求
        elif category == "format_violation":
            visual_description += f"【Format Violation Error】\n"
            visual_description += "Problem Analysis: The model's output format does not comply with the required JSON structure\n"
            visual_description += "Example answer format:\n \
                                    <think>your detailed comparative reasoning and analysis process here</think>\
                                    <answer>Answer with one of the following: \"Yes\" (high confidence anomaly), \"Possible\" (possible anomaly), \"Uncertain\" (uncertain anomaly), or \"No\" (no anomaly). \
                                    If \"Yes\", \"Possible\", or \"Uncertain\", continue to output all locations, the format should be like {'bbox_2d': [x1, y1, x2, y2], 'label': '<anomaly type>'}</answer>."
        
        error['visual_description'] = visual_description
    
    def _draw_detection_boxes(self, error: Dict):
        """在图片上预测框，并保存可视化图片"""
        image_path = error.get('image_path', '')
        
        try:
            # 如果原本图片标签正常读取原始图片，如果原始标签异常读取mask处理好的图片
            if error['defect_type'] == 'good':
                image = resize_image(image_path)
            else:
                try:
                    image = resize_image(os.path.join(DATASET_PROCESSED_DIR, error['class'], error['defect_type'],
                                                  'masked_pictures', error['image_name']))
                except:
                    print(f"无法读取图片: {os.path.join(DATASET_PROCESSED_DIR, error['class'], error['defect_type'], 'masked_pictures', error['image_name'])}\
                          请先运行图片处理程序！")
                    return
            
            # 如果有标注框绘制标注框否则直接返回
            vis_image = np.array(image).copy()
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            height, width = vis_image.shape[:2]
            processed_image_path = f"./temp_image/temp_processed_image/{error['class']}/{error['category']}"
            os.makedirs(processed_image_path, exist_ok=True)
            error['processed_image_path'] = os.path.join(processed_image_path, error['defect_type']+error['image_name']) # 这里是为了防止图片重名导致重复
            pred_bbox = error.get('predicted_bboxes', [])
            if pred_bbox == []:
                cv2.imwrite(error["processed_image_path"], vis_image)
                return
            
            confidence = error.get('confidence', 0)
            for pred_bbox in error['predicted_bboxes']:
                bbox = pred_bbox.get('bbox_2d', [])
                pred = pred_bbox['label']
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    # 绘制边界框
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width-1, x2), min(height-1, y2)
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色框，线宽2
                    
                    # 添加预测标签和置信度
                    label_text = f"{pred} {confidence:.1f}"
                    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_PLAIN, 0.5, 1)[0]
                    cv2.rectangle(vis_image, (x1, y2), 
                                (x1+label_size[0], y2+label_size[1]+4), (0, 0, 255), -1) 
                    cv2.putText(vis_image, label_text, (x1, y2+label_size[1]+2), 
                            cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 1)
            
            cv2.imwrite(error["processed_image_path"], vis_image)
            
            # 将可视化图片路径保存到error中
            print(f"已生成可视化图片: {error["processed_image_path"]}")
            
        except Exception as e:
            print(f"绘制标注框时出错: {e}")
            import traceback
            traceback.print_exc()

    def call_optimizer_llm(self, typical_errors: List[Dict], current_accuracy: Dict, 
                                    iteration: int) -> Dict[str, Any]:
        """调用改进者LLM，包含实际的图片数据, 当前是每一轮优化都会开启一段新的对话, 后续可以考虑使用将不同次的对话使用同一段对话"""
        
        system_prompt = """You are a professional prompt optimization expert. 
                        Your task is to improve prompts to enhance detection accuracy by 
                        analyzing the ADLLM's error cases in anomaly detection tasks.
                        Please analyze the provided images along with their error descriptions 
                        to understand the specific detection failures and propose improvements."""

        # 构建初始信息
        accuracy_text = (
        f"Accuracy: {current_accuracy['accuracy']:.2%}, "
        f"Precision: {current_accuracy['precision']:.2%}, "
        f"Recall: {current_accuracy['recall']:.2%}, "
        f"F1: {current_accuracy['f1']:.2%}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"text": f"Current performance: {accuracy_text}\nCurrent prompt:\n{self.current_prompt}\nPlease analyze the images below and propose prompt improvements."},
            ]}
        ]

        # 添加历史优化记录（如果有的话）
        if self.optimization_history and len(self.optimization_history) > 0:
            history_text = "\n## Previous Optimization History:\n"
            for i, record in enumerate(self.optimization_history[-10:]):
                previous_performance = record['accuracy_before']
                accuracy_text = (
                f"Accuracy: {previous_performance['accuracy']:.2%}, "
                f"Precision: {previous_performance['precision']:.2%}, "
                f"Recall: {previous_performance['recall']:.2%}, "
                f"F1: {previous_performance['f1']:.2%}"
        )
                history_text += f"\n### Iteration {record['iteration']}:\n"
                history_text += f"- Previous Performance: {accuracy_text}\n"
                history_text += f"- Improvement Strategy: {record['improvement_strategy']}\n"
                history_text += f"- Prompt Changes: {len(record['old_prompt'])} -> {len(record['new_prompt'])} characters\n"
            messages[1]['content'].append({"text": history_text})
        messages[1]['content'].append({
            "text": "Please analyze the images below and propose prompt improvements."
        })
        
        # 为每个典型错误添加图片
        for i, error in enumerate(typical_errors):
            if error['category'] == 'format_violation':
                messages[1]['content'].append({
                    "text": f"Case {i+1} - {error['category']}: {error['visual_description']}"
                })
            else:
                try:
                    image_data = encode_image(error['processed_image_path'])
                    messages[1]['content'].extend([{
                        "image": f"data:image/png;base64,{image_data}"},
                        {"text": f"Case {i+1} - {error['category']}: {error['visual_description']}\n \
                        ADLLM's Incorrect Reasoning Process:\n{error['think']}\n"
                    }])
                except Exception as e:
                    print(f"无法读取图片 {error['processed_image_path']}: {e}")

        # 添加格式化输出的要求
        messages[1]['content'].append({
            "text": (
                "Please provide your analysis and improvement strategy in the following XML-like format:\n"
                "<analysis>Detailed analysis of the errors and their causes...</analysis>\n"
                "<improvement_strategy>Specific strategies to improve the prompt...</improvement_strategy>\n"
                "<new_prompt>The complete revised prompt incorporating the improvements...</new_prompt>\n"
                "CRITICAL CONSTRAINTS:\n"
                "1. DO NOT modify ADLLM's output format structure\n"
                "2. The <think> and <answer> tags must remain unchanged\n"
                "3. Keep the exact JSON bounding box format: [{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"anomaly_type\"}, ...]\n"
                "4. Maintain the same response options: \"Yes\", \"Possible\", \"Uncertain\", \"No\"\n"
                "5. Preserve all existing output format requirements\n\n"
                "Ensure the new prompt maintains clarity and conciseness while enhancing detection capabilities."
            )
        })
        
        max_retries = 5

        # 记录调用的messages，方便调试
        with open('message.json', 'w', encoding='utf-8') as img_file:
            json.dump(messages, img_file, ensure_ascii=False, indent=4)
        
        for attempt in range(max_retries):
            response = dashscope.MultiModalConversation.call(
                model='qwen3-vl-plus', 
                messages=messages,
                top_p=0.9,
                temperature=0,
                seed=42
            )
            
            if response.status_code == HTTPStatus.OK:
                text_content = ""
                for content in response.output.choices[0].message.content:
                    if 'text' in content:
                        text_content += content['text']

                # 解析LLM回答
                analysis_match = re.search(r'<analysis>(.*?)</analysis>', text_content, re.DOTALL)
                strategy_match = re.search(r'<improvement_strategy>(.*?)</improvement_strategy>', text_content, re.DOTALL)
                prompt_match = re.search(r'<new_prompt>(.*?)</new_prompt>', text_content, re.DOTALL)
                analysis = analysis_match.group(1).strip() if analysis_match else "No analysis"
                strategy = strategy_match.group(1).strip() if strategy_match else "No strategy"
                new_prompt = prompt_match.group(1).strip() if prompt_match else self.current_prompt
                
                return {
                    "analysis": analysis,
                    "strategy": strategy,
                    "new_prompt": new_prompt,
                    "raw_response": text_content
                }
            else:
                # 打印具体的API错误信息
                print(f"API调用失败 - 状态码: {response.status_code}, 错误代码: {response.code}, 错误信息: {response.message}")
                if hasattr(response, 'request_id'):
                    print(f"请求ID: {response.request_id}")
        
        return {
            "analysis": "API call failed",
            "strategy": "Keep original prompt",
            "new_prompt": self.current_prompt,
            "raw_response": "API call failed"
        }
    
    def optimize_prompt(self, error_results: List[Dict], current_accuracy: Dict, iteration: int) -> Dict[str, Any]:
        """执行一次prompt优化迭代"""
        print(f"\n开始第 {iteration} 轮prompt优化...")
        
        error_categories = self.classify_errors(error_results)
        print("错误分类完成")

        typical_errors = self.select_typical_errors(error_categories)
        print(f"选择了 {len(typical_errors)} 个典型错误案例")
        
        optimization_result = self.call_optimizer_llm(typical_errors, current_accuracy, iteration)
        print("调用优化者LLM完成")
        
        old_prompt = self.current_prompt
        self.current_prompt = optimization_result['new_prompt']
        
        optimization_record = {
            "iteration": iteration,
            "old_prompt": old_prompt,
            "new_prompt": self.current_prompt,
            "accuracy_before": current_accuracy,
            "typical_errors": typical_errors,
            "optimizer_analysis": optimization_result['analysis'],
            "improvement_strategy": optimization_result['strategy'],
            "timestamp": time.time()
        }
        
        self.optimization_history.append(optimization_record)
        
        print(f"第 {iteration} 轮优化完成")
        print(f"改进策略: {optimization_result['strategy'][:100]}...")
        
        return optimization_record

def run_automated_prompt_optimization(
    base_prompt: str, 
    api_key: str, 
    num_iterations: int = 1000,
    test_function: callable = None,
    save_path: str = ""
):
    """运行自动化的prompt优化流程"""
    
    optimizer = PromptOptimizer(base_prompt, api_key, save_path=save_path)
    
    for iteration in range(1, num_iterations + 1):
        print(f"\n{'='*60}")
        print(f"开始第 {iteration}/{num_iterations} 轮优化迭代")
        print(f"{'='*60}")
        
        # 使用当前prompt进行测试
        current_accuracy, error_results = test_function(optimizer.current_prompt)
        stop_bool, reason = optimizer.early_stopping.check_early_stopping(current_accuracy, iteration, optimizer.current_prompt)
        print(reason)
        if stop_bool:
            break
        # 如果没有错误（活在梦里），提前结束
        if len(error_results) == 0:
            print("没有错误样本，优化完成！")
            break

        # 这里是为了debug测试一下
        # current_accuracy = {
        #                     "accuracy": 0.8235,
        #                     "precision": 0.7812,
        #                     "recall": 0.7143,
        #                     "f1": 0.7463,
        #                     "false_positive_rate": 0.1250,
        #                     "confusion_matrix": [[45, 8], [12, 35]]
        #                     }
        # error_results = json.load(open("LLMtry/72b-plus/mvtec/breakfast_box_detailed_results.json", 'r', encoding='cp1252'))['results']
        # stop_bool, reason = optimizer.early_stopping.check_early_stopping(current_accuracy, iteration, optimizer.current_prompt)
        # if stop_bool:
        #     break
        # # 如果没有错误（活在梦里），提前结束
        # if len(error_results) == 0:
        #     print("没有错误样本，优化完成！")
        #     break

        print(f"当前结果: {json.dumps(current_accuracy, indent=4)}")
        print(f"错误样本数: {len(error_results)}")
        
        # 执行prompt优化
        optimization_record = optimizer.optimize_prompt(error_results, current_accuracy, iteration)
        
        # 保存本轮优化结果
        save_optimization_result(optimization_record, iteration)
        
        print(f"第 {iteration} 轮优化完成，准备下一轮...")
        
        # 等待一段时间避免api调用过于频繁
        time.sleep(3)
    
    # 生成最终优化总结
    final_summary = generate_final_summary(optimizer.optimization_history)
    print("\n优化完成！")
    print(final_summary)
    
    return optimizer.current_prompt, final_summary

def save_optimization_result(record: Dict, iteration: int):
    """保存优化结果"""
    os.makedirs(f"./log/{DATASET}/{SELECTED_CLASS}/optimization_results", exist_ok=True)
    
    filename = f"./log/{DATASET}/{SELECTED_CLASS}/optimization_results/iteration_{iteration}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    
    print(f"优化结果已保存到: {filename}")

def generate_final_summary(history: List[Dict]) -> str:
    """生成最终优化总结"""
    summary = "Prompt优化过程总结\n"
    summary += "=" * 50 + "\n\n"
    
    for record in history:
        summary += f"第 {record['iteration']} 轮优化:\n"
        summary += f"  准确率: {record['accuracy_before']}\n"
        summary += f"  主要改进: {record['improvement_strategy']}\n"
        summary += f"  提示词变化: {len(record['old_prompt'])} -> {len(record['new_prompt'])} 字符\n\n"
    
    return summary

# 将文件的print输入到log文件当中
class TeeOutput:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush() 
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()

def setup_logging(log_path: str):
    '''设置日志记录, 同时输出到控制台和日志文件'''
    os.makedirs(log_path, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_path, f"optimization_{timestamp}.log")
    
    tee = TeeOutput(log_file)
    sys.stdout = tee
    sys.stderr = tee
    
    print(f"日志文件: {log_file}")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    return tee  # 返回对象以便后续关闭

if __name__ == "__main__":
    #全局配置
    os.environ["DASHSCOPE_API_KEY"] = "sk-1e6d5dd3c6a94151ab15f67cd0b281a8"
    BASE_DIR = "./LLM_prompt"
    DATASET = "mvtec_loco_anomaly_detection"
    DATASET_DIR = f"./dataset/{DATASET}" 
    ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations", "mvtec")
    SELECTED_CLASS = [d for d in os.listdir(DATASET_DIR) 
                if os.path.isdir(os.path.join(DATASET_DIR, d))][0]
    DATASET_PROCESSED_DIR = f"./dataset_masked/{DATASET}"

    base_prompt = (
        "The first image is a normal reference sample. \n"
        "The second image is a test image that may contain defects. Carefully compare the two images to \
        determine if there are any anomalies in the test image.\n\n"
        "Output format requirements:\n"
        "1. First, provide detailed reasoning in <think> tags\n"
        "2. Then, provide your final answer in <answer> tags\n"
        "3. If no anomalies found, answer: 'No'\n"
        "4. If anomalies found, answer: 'Yes' or 'Possible' or 'Uncertain' followed by a list of bounding boxes in JSON format:\n"
        "   [{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"anomaly_type\"}, ...]([x1, y1, x2, y2] format)\n\n"
        "Example answer format:\n"
        "<think>your detailed comparative reasoning and analysis process here</think>\
        <answer>Answer with one of the following: \"Yes\" (high confidence anomaly), \"Possible\" (possible anomaly), \"Uncertain\" (uncertain anomaly), or \"No\" (no anomaly). \
        If \"Yes\", \"Possible\", or \"Uncertain\", continue to output all locations, the format should be like {'bbox_2d': [x1, y1, x2, y2], 'label': '<anomaly type>'}</answer>."
    )
    
    api_key = "sk-1e6d5dd3c6a94151ab15f67cd0b281a8"

    # 这里传进去的时候是针对一个数据集的其中的一个类别，针对不同的类别需要不同的初始化
    prompt_tester = prompt_AD_tester(SELECTED_CLASS, BASE_DIR, DATASET_DIR, ANNOTATIONS_DIR, api_key)
    log_path = f"./log/{DATASET}/{SELECTED_CLASS}/text"
    png_log_path = f"./log/{DATASET}/{SELECTED_CLASS}/figures"
    setup_logging(log_path) # 设置日志记录
    optimized_prompt, summary = run_automated_prompt_optimization(
        base_prompt=base_prompt,
        api_key=api_key,
        num_iterations=100,
        test_function=prompt_tester.prompt_AD,
        save_path=os.path.join(png_log_path, "record.png")
    )

    
    print("Prompt优化系统设计完成！")
    print("要使用此系统，你需要：")
    print("1. 修改你的process_class函数，使其接受prompt参数")
    print("2. 实现测试适配器函数")
    print("3. 配置适当的API密钥")