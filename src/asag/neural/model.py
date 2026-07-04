"""DeBERTa cross-encoder grader: a transformer backbone + a task head.

The backbone is any ``AutoModel`` (default ``microsoft/deberta-v3-base``); the
head depends on ``task_type``:

* regression     → ``Linear(h, 1)``        (raw output, MSE)
* classification → ``Linear(h, n_classes)`` (logits, cross-entropy)
* ordinal        → CORAL / CORN head over ``num_classes`` ranks

Pooling is the ``[CLS]`` token of the last hidden state (DeBERTa has no pooler in
``AutoModel``) or mean-pooling over the attention mask. ``forward`` returns raw
head outputs; loss/prediction live in :mod:`asag.neural.heads` and the trainer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from asag.neural.heads import CoralHead


class CrossEncoderGrader(nn.Module):
    def __init__(self, backbone: str, task_type: str, *, num_classes: int = 0,
                 dropout: float = 0.1, pooling: str = "cls",
                 ordinal_head: str = "corn",
                 freeze_backbone: bool = False, lora_enabled: bool = False,
                 lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1):
        super().__init__()
        self.task_type = task_type
        self.pooling = pooling
        self.ordinal_head = ordinal_head
        self.num_classes = num_classes
        # Force fp32: newer transformers (5.x) load the deberta-v3 checkpoint in its
        # native fp16, but the task head below is fp32 — mixing them raises
        # "mat1 and mat2 must have the same dtype (Half vs Float)". .float() is a
        # no-op on the older fp32-loading transformers, so it is safe either way.
        self.encoder = AutoModel.from_pretrained(backbone).float()
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        # Overfitting controls for small corpora (configured via NeuralCfg).
        # LoRA wins over a plain freeze if both are set.
        if lora_enabled:
            from peft import LoraConfig, get_peft_model
            self.encoder = get_peft_model(self.encoder, LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                bias="none", target_modules=["query_proj", "key_proj", "value_proj"],
            ))
        elif freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        if task_type == "regression":
            self.head: nn.Module = nn.Linear(h, 1)
        elif task_type == "classification":
            if num_classes < 2:
                raise ValueError("classification needs num_classes >= 2")
            self.head = nn.Linear(h, num_classes)
        elif task_type == "ordinal":
            if ordinal_head == "coral":
                self.head = CoralHead(h, num_classes)
            else:  # corn: K-1 independent logits
                self.head = nn.Linear(h, num_classes - 1)
        else:
            raise ValueError(f"unknown task_type {task_type!r}")

    def _pool(self, last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            m = mask.unsqueeze(-1).float()
            return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return last_hidden[:, 0]                      # [CLS]

    def forward(self, input_ids, attention_mask, **kw) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           **{k: v for k, v in kw.items() if k == "token_type_ids"})
        pooled = self.dropout(self._pool(out.last_hidden_state, attention_mask))
        z = self.head(pooled)
        return z.squeeze(-1) if self.task_type == "regression" else z
