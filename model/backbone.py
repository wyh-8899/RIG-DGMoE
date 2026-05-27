import logging
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from peft import LoraConfig, TaskType, get_peft_model


class Backbone(nn.Module):
    """
    GPT2 + LoRA backbone
    只负责:
      1) 接收 inputs_embeds
      2) 输出 hidden states
    """
    def __init__(
        self,
        model_path: str = "./gpt2",
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ):
        super().__init__()

        self.model_path = model_path
        self.gpt2_config = GPT2Config.from_pretrained(model_path)

        self.lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["c_attn", "c_proj", "c_fc"],
            fan_in_fan_out=True,
        )

        base_model = GPT2Model.from_pretrained(
            model_path,
            config=self.gpt2_config,
            local_files_only=True,
        )
        self.gpt2 = get_peft_model(base_model, self.lora_config)

        logging.info("Backbone initialized with GPT2 + LoRA")

    @property
    def d_model(self) -> int:
        return self.gpt2_config.n_embd

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return {
            "hidden_states": outputs.last_hidden_state,   # [B, L, D]
        }