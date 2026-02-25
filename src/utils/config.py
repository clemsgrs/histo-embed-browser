from omegaconf import OmegaConf

from src.configs import default_config


def get_cfg_from_file(config_file):
    default_cfg = OmegaConf.create(default_config)
    cfg = OmegaConf.load(config_file)
    cfg = OmegaConf.merge(default_cfg, cfg)
    OmegaConf.resolve(cfg)
    return cfg
