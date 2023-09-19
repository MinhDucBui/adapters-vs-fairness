from src.module.base_module import BaseModule
from omegaconf import DictConfig
from src.utils import utils
import hydra
import torch.nn as nn
import torch
from typing import Any, Callable, NamedTuple, Optional, Union
import torch.nn.functional as F


log = utils.get_logger(__name__)



class JigsawClassification(BaseModule):
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        
        super().__init__(*args, **kwargs)
        
        
    def forward(self, batch, *args, **kwargs):
        # MOVE TO MODEL
        outputs = self.model.forward(
            input_ids=batch["input_ids"],
            #token_type_ids=batch["token_type_ids"].squeeze(),
            attention_mask=batch["attention_mask"],
            #labels=batch["labels"],
        )
        return outputs
    
    
    def training_step(self, batch: dict, batch_idx: int) -> dict[str, Any]:
        """Comprises training step of your model which takes a forward pass.

        **Notes:**
            If you want to extend `training_step`, add a `on_train_batch_end` method via overrides.
            See: Pytorch-Lightning's `on_train_batch_end <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#on-train-batch-end>`_

        **Implementation:**

            .. code-block:: python

                def training_step(
                    self, batch: BatchEncoding, batch_idx: int
                ) -> Union[dict[str, Any], ModelOutput]:
                    outputs = self(batch)
                    self.log("train/loss", outputs.loss)
                    return outputs

        Args:
            batch: typically comprising input_ids, attention_mask, and position_ids
            batch_idx: variable used internally by pytorch-lightning

        Returns:
            Union[dict[str, Any], ModelOutput]: model output that must have 'loss' as attr or key
        """
        outputs = self(batch)
        # Calculate the binary cross-entropy loss using functional interface
        loss = F.binary_cross_entropy_with_logits(outputs["logits"].squeeze(), batch["labels"])
        self.log("train/loss", loss)
        return {"loss": loss}
        
    def get_preds(self, outputs: dict, *args, **kwargs) -> dict:
        prob = torch.sigmoid(outputs["logits"])
        outputs["preds"] = (prob >= 0.5).int().squeeze()
        return outputs
    
    def prepare_step_outputs(
        self, stage: str, step_outputs: dict, dataset: Optional[str] = None
    ) -> dict:
        return self.get_preds(step_outputs)
    
    def prepare_outputs(
        self, stage: str, step_outputs: dict, dataset: Optional[str] = None
    ) -> dict:
        step_outputs = self.get_preds(step_outputs)
        return step_outputs
    
    def prepare_batch(
        self, stage: str, batch: dict, dataset: Optional[str] = None
    ) -> dict:
        batch["labels_binary"] = (batch["labels"] >= 0.5).int()
        return batch