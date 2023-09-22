import copy


def assert_functions(cfg):
    pass


def assert_datamodule_cfg(data_cfg):
    for key, value in data_cfg.items():
        assert "_target_" in value.keys(), "No _target_ specified in datamodule {}".format(key)


def assert_loss(base_loss, loss_weighting):
    for loss_name, loss in base_loss.items():
        assert loss_name in loss_weighting.keys(), \
            "Loss {} defined but no loss_weighting found in loss cfg".format(loss_name)

    for loss_name, _ in loss_weighting.items():
        assert loss_name in loss_weighting.keys(), \
            "loss_weighting {} defined but not defined in base_loss in loss cfg".format(loss_name)
