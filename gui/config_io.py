import yaml

DEFAULT_CONFIG_PATH = "config/config.yaml"


def load_config(path=DEFAULT_CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(config, path=DEFAULT_CONFIG_PATH):
    with open(path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
