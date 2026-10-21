from torch import optim as optim


def build_optimizer(config, model):
    """
    Builds optimizer for the model from the config.
    """
    optimizer_type = config.TRAIN.OPTIMIZER.NAME.lower()
    lr = config.TRAIN.LR
    momentum = config.TRAIN.OPTIMIZER.MOMENTUM
    eps = config.TRAIN.OPTIMIZER.EPS
    betas = config.TRAIN.OPTIMIZER.BETAS
    wd = config.TRAIN.WEIGHT_DECAY
    parameters = model.parameters()

    if optimizer_type == "sgd":
        optimizer = optim.SGD(parameters, lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
    elif optimizer_type == "adam":
        optimizer = optim.Adam(parameters, lr=lr, eps=eps, betas=betas, weight_decay=wd)
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(parameters, lr=lr, eps=eps, betas=betas, weight_decay=wd)
    else:
        raise NotImplementedError(f"Unknown optimizer: {optimizer_type}")

    return optimizer
