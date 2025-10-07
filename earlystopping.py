import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

class EarlyStopping:
    """这里写一个EarlyStopping类来判断模型运行该何时停止,以及记录最优的prompt和优化过程中的各个指标的变化"""
    
    def __init__(self, patience=10, min_improvement=0, base_prompt="", save_path=""):
        self.patience = patience  
        self.min_improvement = min_improvement  
        self.no_improvement_count = 0
        self.best_prompt = base_prompt
        self.accuracy_history = []
        self.save_path = save_path
        self.best_accuracy = 0

    def check_early_stopping(self, current_accuracy: Dict, iteration: int, current_prompt: str) -> Tuple[bool, str]:
        """检查是否应该提前停止优化"""
        current_acc = current_accuracy.get('accuracy', 0)
        improvement = current_acc - self.best_accuracy
        
        # 记录各项指标并将其画图
        self.accuracy_history.append(current_accuracy)
        self.plot_optimization_history()

        if improvement > self.min_improvement:
            self.best_accuracy = current_acc
            self.best_prompt = current_prompt
            self.no_improvement_count = 0
            return False, f"准确率提升 {improvement:.4f}，继续优化"
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                reason = f"连续 {self.patience} 轮准确率无显著改进（改进: {improvement:.4f} < 阈值: {self.min_improvement:.4f}）"
                return True, reason
            else:
                return False, f"准确率无显著改进 ({improvement:.4f})，计数: {self.no_improvement_count}/{self.patience}"
    
    def plot_optimization_history(self):
        """绘制优化过程指标图表"""
        if len(self.accuracy_history) < 2:
            print("历史数据不足暂时无法绘制图表")
            return
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Prompt Optimization Process History', fontsize=16)
        
        iterations = list(range(1, len(self.accuracy_history) + 1))
        
        # 1. 主要准确率指标
        accuracy_values = [h['accuracy'] for h in self.accuracy_history]
        precision_values = [h['precision'] for h in self.accuracy_history]
        recall_values = [h['recall'] for h in self.accuracy_history]
        f1_values = [h['f1'] for h in self.accuracy_history]
        
        ax1.plot(iterations, accuracy_values, 'o-', label='Accuracy', linewidth=2)
        ax1.plot(iterations, precision_values, 's-', label='Precision', linewidth=2)
        ax1.plot(iterations, recall_values, '^-', label='Recall', linewidth=2)
        ax1.plot(iterations, f1_values, 'd-', label='F1 Score', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Score')
        ax1.set_title('Main Accuracy Metrics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 错误率指标
        fp_rates = [h['false_positive_rate'] for h in self.accuracy_history]
        fn_rates = [1 - h['recall'] for h in self.accuracy_history]  # 假阴性率
        
        ax2.plot(iterations, fp_rates, 'o-', label='False Positive Rate', color='red', linewidth=2)
        ax2.plot(iterations, fn_rates, 's-', label='False Negative Rate', color='orange', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Rate')
        ax2.set_title('Error Rates')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 改进趋势
        accuracy_improvements = [0] + [accuracy_values[i] - accuracy_values[i-1] for i in range(1, len(accuracy_values))]
        
        ax3.bar(iterations, accuracy_improvements, color=['green' if x >= 0 else 'red' for x in accuracy_improvements])
        ax3.axhline(y=self.min_improvement, color='blue', linestyle='--', label=f'Min Improvement ({self.min_improvement:.3f})')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Accuracy Improvement')
        ax3.set_title('Accuracy Improvement per Iteration')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 混淆矩阵变化（最后一个迭代）
        if 'confusion_matrix' in self.accuracy_history[-1]:
            cm = self.accuracy_history[-1]['confusion_matrix']
            cm_matrix = np.array(cm)
            im = ax4.imshow(cm_matrix, cmap='Blues', interpolation='nearest')
            
            # 添加数值标签
            for i in range(cm_matrix.shape[0]):
                for j in range(cm_matrix.shape[1]):
                    ax4.text(j, i, str(cm_matrix[i, j]), 
                            ha="center", va="center", 
                            color="white" if cm_matrix[i, j] > cm_matrix.max()/2 else "black")
            
            ax4.set_xlabel('Predicted')
            ax4.set_ylabel('Actual')
            ax4.set_title('Latest Confusion Matrix')
            ax4.set_xticks([0, 1])
            ax4.set_yticks([0, 1])
            ax4.set_xticklabels(['Normal', 'Anomaly'])
            ax4.set_yticklabels(['Normal', 'Anomaly'])
            plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        
        # 保存图表
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"优化历史图表已保存到: {self.save_path}")

    