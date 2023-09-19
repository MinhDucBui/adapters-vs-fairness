from types import MethodType
from typing import Any, Callable, Optional, Union
import copy
import pytorch_lightning as pl
from src.module.mixin.optimizer import OptimizerMixin
from src.module.mixin.eval import EvalMixin
#from src.module.mixin.initialize_models import InitializeModelsMixin
import torch
from omegaconf import DictConfig
from src.utils import utils
from src.models.modules.utils import change_embedding_layer
from src.utils.utils import keep_only_model_forward_arguments, get_model_language, \
    append_torch_in_dict, get_subset_cleaned_batch
from src.utils.assert_functions import assert_functions
from src.utils.debug import debug_embedding_updating
from torch.nn import functional as F
log = utils.get_logger(__name__)
import hydra
            
            
#class BaseModule(OptimizerMixin, EvalMixin, InitializeModelsMixin, pl.LightningModule):
class BaseModule(OptimizerMixin, EvalMixin, pl.LightningModule):
    def __init__(self,         
                 model: DictConfig,
                 optimizer: DictConfig,
                 scheduler: Optional[DictConfig],
                 evaluation: Optional[DictConfig] = None,        
                 *args: Any,
                 **kwargs: Any,):
        """method used to define our model parameters"""
        # Sanity Check Config
        #assert_functions(copy.deepcopy(cfg))

        self.save_hyperparameters()
        #self.cfg = cfg
        #self.data_cfg = cfg.datamodule
        #self.trainer_cfg = cfg.trainer

        pl.LightningModule.__init__(self)
        super().__init__()
        
        # Lightning 2.0 requires manual management of evaluation step outputs
        self._eval_outputs = []
        
        
    def setup(self, stage: str):
        """Sets up the TridentModule.

        Setup the model if it does not exist yet. This enables inter-operability between your datamodule and model if you pass define a setup function in `module_cfg`, as the datamodule will be set up _before_ the model.

        In case you pass : :obj:`setup` to :obj:`module_cfg` the function should follow the below schema:

            .. code-block:: python

                def setup(module: TridentModule, stage: str):
                    # the module has to be setup
                    module.model = hydra.utils.instantiate(module.hparams.model)

        Since the :obj:`module` exposes :obj:`module.trainer.datamodule`, you can use your custom setup function to enable inter-operability between the module and datamodule.

        """
        # TODO: maybe we can simplify and integrate this even better
        if hasattr(self.hparams, "model"):
            self.model = hydra.utils.instantiate(self.hparams.model)
        else:
            raise ValueError("Model not specified in self.hparams. Please provide a model.")

        if ckpt := getattr(self.hparams, "weights_from_checkpoint", None):
            self.weights_from_checkpoint(**ckpt)
    

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
        self.log("train/loss", outputs["loss"])
        return outputs


    def test_step(self, batch, batch_idx):
        preds, loss, acc = self._get_preds_loss_accuracy(batch)
        self.log("test_loss", loss)
        self.log("test_accuracy", acc)
        return {"loss": loss, "acc": acc}

    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar"""
        output = self.model.forward(
            input_ids=batch["input_ids"],
            #token_type_ids=batch["token_type_ids"].squeeze(),
            attention_mask=batch["attention_mask"],
            #labels=batch["labels"],
        )
        # Assuming you have a binary tensor of labels
        logits = output.logits  # Assuming the logits are of shape (batch_size, 1)

        # Reshape the labels to match the shape of logits
        labels = batch["labels"].float().unsqueeze(1)  # Shape: (batch_size, 1)
  
        # Calculate the binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')

        #preds = torch.argmax(output.logits, dim=1)
        #loss = output.loss.mean()
        #acc = self.accuracy.compute(
        #    references=batch["labels"].data, predictions=preds.data
        #)
        return 0, loss, 0
