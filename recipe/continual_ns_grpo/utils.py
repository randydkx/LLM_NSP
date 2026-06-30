import re
import torch.nn as nn
from typing import Optional, List

MLP_TARGETS = ["mlp.down_proj", "mlp.gate_proj", "mlp.up_proj"]
ATTN_TARGETS = ["self_attn.k_proj", "self_attn.o_proj", "self_attn.q_proj", "self_attn.v_proj"]

class QwenModuleEligibilityChecker:
    def __init__(
        self,
        start_MLP: Optional[int] = None,
        end_MLP: Optional[int] = None,
        start_attention: Optional[int] = None,
        end_attention: Optional[int] = None,
        mlp_targets: List[str] = MLP_TARGETS,
        attn_targets: List[str] = ATTN_TARGETS,
    ):
        """
        针对 Qwen2.5-7B 结构的模块筛选器
        
        Args:
            start_MLP, end_MLP: MLP 投影层适用的层号闭区间（含端点）；None 表示不启用
            start_attention, end_attention: Attention 投影层适用的层号闭区间（含端点）；None 表示不启用
            mlp_targets: MLP 模块的目标子模块名称列表
            attn_targets: Attention 模块的目标子模块名称列表
        """
        self.start_MLP = start_MLP
        self.end_MLP = end_MLP
        self.start_attention = start_attention
        self.end_attention = end_attention
        self.mlp_targets = mlp_targets
        self.attn_targets = attn_targets
        
        # 编译正则表达式来提取层号
        self.layer_pattern = re.compile(r"model\.layers\.(\d+)\.")

    def _extract_layer_index(self, module_name: str) -> Optional[int]:
        """
        从模块名中提取层号
        """
        match = self.layer_pattern.search(module_name)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    def _in_range(self, idx: int, start: Optional[int], end: Optional[int]) -> bool:
        """
        检查层号是否在指定范围内
        """
        if start is None or end is None:
            return False
        return start <= idx <= end

    def check(self, module_name: str, module) -> bool:
        """
        检查模块是否符合零空间投影的条件，主要是判断是否指定的MLP层以及Attention层中的module，并且是否包含Linear层
        
        Args:
            module_name: 模块的完整名称
            module: 模块对象
            
        Returns:
            bool: 是否符合条件
        """
        # 必须是线性层且有 weight（忽略 bias 与 LayerNorm 等）
        if not (isinstance(module, nn.Linear) and hasattr(module, 'weight')):
            return False
            
        # 提取层号
        layer_idx = self._extract_layer_index(module_name)
        if layer_idx is None:
            return False
            
        # 检查是否为 MLP 目标模块，start_MLP 和 end_MLP 都为 None 则表示不启用 MLP 层的区间筛选
        for mlp_target in self.mlp_targets:
            if mlp_target in module_name and self._in_range(layer_idx, self.start_MLP, self.end_MLP):
                return True
                
        # 检查是否为 Attention 目标模块
        for attn_target in self.attn_targets:
            if attn_target in module_name and self._in_range(layer_idx, self.start_attention, self.end_attention):
                return True
                
        return False