import json
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Hello, World!")
    print(cfg)
    with open(cfg.dataset.folds.file_path, "r") as f:
        splits = json.load(f)
    yaml = OmegaConf.to_yaml(cfg.dataset, resolve=True, sort_keys=True)
    print(yaml)
    print(f"Loaded {len(splits)} splits from {cfg.dataset.folds.file_path}")


if __name__ == "__main__":
    main()
