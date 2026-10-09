import os
import yaml
import __main__

class ConfigManagerPlain:
    """Flat configuration manager (all keys stored at the root of the YAML file)."""
    def __init__(self, config_path="configs/config.yaml"):
        self.config_path = config_path
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        self._data = {}

    def load(self, constants: dict, cli_args: dict = None):
        cli_args = cli_args or {}
        active_cli = {k: v for k, v in cli_args.items() if v is not None}
        updated = False

        if not os.path.exists(self.config_path):
            self._data = {**constants, **active_cli}
            updated = True
        else:
            with open(self.config_path, "r") as f:
                self._data = yaml.safe_load(f) or {}

            for key, default_val in constants.items():
                if key not in self._data:
                    self._data[key] = default_val
                    updated = True

        for key, cli_val in active_cli.items():
            if self._data.get(key) != cli_val:
                self._data[key] = cli_val
                updated = True

        if updated:
            self.save()

        return self

    def save(self):
        with open(self.config_path, "w") as f:
            yaml.safe_dump(self._data, f)

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'ConfigManagerPlain' object has no attribute '{name}'")

    def to_dict(self):
        return self._data


class ConfigManager:
    """Nested configuration manager (automatically groups keys under the running script's name)."""
    def __init__(self, config_path="configs/config.yaml", namespace: str = None):
        self.config_path = config_path
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        if namespace is None:
            if hasattr(__main__, "__file__") and __main__.__file__:
                self.namespace = os.path.splitext(os.path.basename(__main__.__file__))[0]
            else:
                self.namespace = "default"
        else:
            self.namespace = namespace
            
        self._data = {}

    def load(self, constants: dict, cli_args: dict = None):
        cli_args = cli_args or {}
        active_cli = {k: v for k, v in cli_args.items() if v is not None}
        updated = False

        all_configs = {}
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                all_configs = yaml.safe_load(f) or {}

        if self.namespace not in all_configs:
            all_configs[self.namespace] = {**constants, **active_cli}
            updated = True

        self._data = all_configs[self.namespace]

        for key, default_val in constants.items():
            if key not in self._data:
                self._data[key] = default_val
                updated = True

        for key, cli_val in active_cli.items():
            if self._data.get(key) != cli_val:
                self._data[key] = cli_val
                updated = True

        all_configs[self.namespace] = self._data

        if updated:
            self.save(all_configs)

        return self

    def save(self, all_configs=None):
        if all_configs is None:
            all_configs = {}
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    all_configs = yaml.safe_load(f) or {}
            all_configs[self.namespace] = self._data

        with open(self.config_path, "w") as f:
            yaml.safe_dump(all_configs, f)

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'ConfigManager' object has no attribute '{name}'")

    def to_dict(self):
        return self._data


