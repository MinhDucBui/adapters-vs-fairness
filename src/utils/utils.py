import logging
import sys
import warnings
from typing import List, Sequence
from lightning.pytorch.loggers import Logger
import lightning.pytorch as pl
import rich.syntax
import rich.tree
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only


def flatten_dict(inputs: list[dict]) -> dict:
    """Conflates keys of list[dict] and stacks np arrays & tensors along 0-dim."""
    ret = {}
    for input_ in inputs:
        for k, v in input_.items():
            if k not in ret:
                ret[k] = []
            ret[k].append(v)
    for k, v in ret.items():
        if isinstance(v[0], torch.Tensor):
            dim = v[0].dim()
            # stack matrices along first axis
            if dim == 2:
                ret[k] = stack_or_pad_2d(v)
            # concatenate vectors
            elif dim == 1:
                ret[k] = torch.cat(v, dim=0)
            # pad varying dimension and concatenate
            elif dim == 3:
                ret[k] = concatenate_3d(v)
            else:
                raise NotImplementedError(
                    f"Handling {dim} number of dimensions unimplemented"
                )
        elif isinstance(v[0], np.ndarray):
            ret[k] = np.vstack(v)
        else:
            pass
    return ret


def convert_cfg_tuple(cfg_tuple):
    if not ("(" == cfg_tuple[0] and ")" == cfg_tuple[-1]):
        sys.exit("Given cfg_tuple {} is not a tuple. Please make sure that its a tuple.".format(cfg_tuple))
    new_tuple = []
    cfg_tuple = cfg_tuple.replace(" ", "")
    splitted_tuple = tuple(map(str, cfg_tuple.strip('()').split('),(')))
    for single_tuple in splitted_tuple:
        single_tuple = tuple(map(str, single_tuple.strip('()').split(',')))
        new_tuple.append(single_tuple)

    return tuple(new_tuple)


def append_torch_in_dict(dict_to_add, dict_to_extend):
    for key, value in dict_to_add.items():
        if key not in dict_to_extend.keys():
            dict_to_extend[key] = dict_to_add[key]
        else:
            dict_to_extend[key] = torch.cat((dict_to_extend[key], dict_to_add[key]), dim=0)
    return dict_to_extend


def stack_or_pad_2d(tensors: list[torch.Tensor], pad_id=-100) -> torch.Tensor:
    """
    Stack along first axis of latter axis is homogenous in length else pad and stack.
    """
    N, D = zip(*[tuple(x.shape) for x in tensors])
    if len(set(D)) != 1:
        out = torch.full_like(
            torch.Tensor(sum(N), max(D)), fill_value=-100, device=tensors[0].device
        )
        start = 0
        for t in tensors:
            num, len_ = t.shape
            out[start : start + num, :len_] = t
            start += num
        return out
    return torch.vstack(tensors)


def keep_only_model_forward_arguments(model, batch, remove_additional_keys=None):
    if remove_additional_keys is None:
        remove_additional_keys = []
    cleaned_batch = {key: value for key, value in batch.items() if key not in remove_additional_keys}
    cleaned_batch = {key: value for (key, value) in cleaned_batch.items() if
                     key in model.forward.__code__.co_varnames}

    return cleaned_batch


def get_subset_dict(full_set: dict, idx: torch.Tensor):
    """Get subset of batches that are contained in a dictionary.

    Args:
        full_set:
        idx:

    Returns:

    """
    subset = {}
    for key, value in full_set.items():
        # TODO: Fix for now. Wait for Marlena
        if key == "hidden_states" or key == "attentions":
            continue
        subset[key] = value[idx]

    return subset


# Utils for Hydra
def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
        config: DictConfig,
        fields: Sequence[str] = (
                #"model",
                #"evaluation",
                "trainer",
                "module",
                "datamodule",
                "callbacks",
                "logger",
                "seed",
        ),
        resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List[Logger],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config.trainer
    hparams["module"] = config["module"]
    hparams["datamodule"] = config["datamodule"]
    
    if "evaluation" in config:
        hparams["evaluation"] = config["evaluation"]
    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    # hparams["student/params_total"] = sum(p.numel() for p in model.parameters())
    # hparams["student/params_trainable"] = sum(
    #    p.numel() for p in model.parameters() if p.requires_grad
    # )
    # hparams["student/params_not_trainable"] = sum(
    #    p.numel() for p in model.parameters() if not p.requires_grad
    # )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def finish(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List[Logger],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        #if isinstance(lg, pl.loggers.wandb.WandbLogger):
        import wandb
        wandb.finish()
