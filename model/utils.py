import torch
import logging
import json

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__
    

def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def restore_bn_statistics(model, bn_statistics):
    """Restore batchnorm running mean and variance from a list of values.

    Args:
        model (torch.nn.Module): Model containing batchnorm modules.
        bn_statistics (list): List of tuples (running_mean, running_var) containing
            the batchnorm running mean and variance to restore.
    """
    idx = 0
    for module in model.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module.running_mean, module.running_var = bn_statistics[idx]
            idx += 1


def save_and_reset_bn_statistics(model):
    """Save batchnorm running mean and variance in a list and reset them to None.

    Args:
        model (torch.nn.Module): Model containing batchnorm modules.
    Returns:
        bn_statistics (list): List of tuples (running_mean, running_var) containing
            the batchnorm running mean and variance to restore.
    """
    bn_statistics = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            # Save current statistics
            current_statistics = (
                module.running_mean.clone() if module.running_mean is not None else None,
                module.running_var.clone() if module.running_var is not None else None
            )
            bn_statistics.append(current_statistics)
            # Reset statistics
            module.running_mean = None
            module.running_var = None
    return bn_statistics


def save_checkpoint(state, filename="mycheckpoint.pth.tar"):
    torch.save(state, filename)


def load_checkpoint(model, optimizer, checkpoint, scheduler):
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return checkpoint['epoch']


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)