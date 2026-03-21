import torch
import copy
import numpy as np

class EarlyStopping:
    """
    高度泛化的 PyTorch 早停机制。
    """
    def __init__(self, patience=7, mode='min', min_delta=1e-4, filepath=None, verbose=False):
        """
        :param patience: 容忍多少个 epoch 验证指标没有提升
        :param mode: 'min' 监控 Loss 等越小越好的指标; 'max' 监控 AUC/ACC 等越大越好的指标
        :param min_delta: 判定为有效提升的最小变化量 (防止微小震荡)
        :param filepath: 若提供路径，则将最佳权重保存至本地 (如 'best_model.pth')
        :param verbose: 是否打印早停的提示信息
        """
        assert mode in ['min', 'max'], "mode 必须是 'min' 或 'max'"
        
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.filepath = filepath
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
        
        # 记录真实指标，方便打印
        self.val_score = np.inf if mode == 'min' else -np.inf

    def __call__(self, current_score, model=None):
        """
        每次 epoch 结束时调用此方法。
        :param current_score: 当前 epoch 的验证指标 (如 val_loss 或 val_auc)
        :param model: 当前的 PyTorch 模型实例 (用于保存权重)
        :return: bool, 是否触发早停
        """
        if self.best_score is None:
            self.best_score = current_score
            self._save_checkpoint(current_score, model)
            return False

        # 判断当前 score 是否优于 best_score
        if self.mode == 'min':
            is_better = current_score < self.best_score - self.min_delta
        else:
            is_better = current_score > self.best_score + self.min_delta

        if is_better:
            self.best_score = current_score
            self._save_checkpoint(current_score, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] 计数器: {self.counter} / {self.patience} (当前最优: {self.val_score:.6f})")
            
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _save_checkpoint(self, current_score, model):
        """当验证指标改善时，保存模型权重"""
        if self.verbose:
            print(f"[EarlyStopping] 验证指标改善 ({self.val_score:.6f} --> {current_score:.6f})。保存模型...")
        self.val_score = current_score
        
        if model is not None:
            # 1. 保存在内存中 (深拷贝，防止随训练被覆盖)
            self.best_weights = copy.deepcopy(model.state_dict())
            
            # 2. 如果指定了路径，则同时落盘
            if self.filepath:
                torch.save(self.best_weights, self.filepath)

    def load_best_weights(self, model):
        """训练结束后，调用此方法将模型恢复到历史最佳状态"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                print("[EarlyStopping] 已成功恢复最佳模型权重。")
        elif self.filepath:
            model.load_state_dict(torch.load(self.filepath))
            if self.verbose:
                print(f"[EarlyStopping] 已从 {self.filepath} 加载最佳模型权重。")
        return model