from dataclasses import dataclass, field, fields
import argparse


def safe_issubclass(son_cls, parent_cls):
    if not isinstance(son_cls, type) or not isinstance(parent_cls, type):
        return False
    return issubclass(son_cls, CLIConfig)


class CLIConfig:
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace, prefix: str="") -> 'CLIConfig':
        """
        1. parse sub config from cli args
        2. parse current config from cli args
        """
        sub_configs_name_value_pairs = cls.sub_configs_from_cli_args(cls, args, prefix)
        curr_config_name_value_pairs = cls.curr_config_from_cli_args(cls, args, prefix)
        return cls(**sub_configs_name_value_pairs, **curr_config_name_value_pairs)

    @staticmethod
    def curr_config_from_cli_args(cls, args: argparse.Namespace, prefix: str="") -> dict:
        name_value_pairs = {}
        for attr in fields(cls):
            name = attr.name
            sub_config_cls = attr.type
            if not safe_issubclass(sub_config_cls, CLIConfig):
                value = getattr(args, prefix+name, getattr(cls, name))
                name_value_pairs[name] = value
        return name_value_pairs

    @staticmethod
    def sub_configs_from_cli_args(cls, args: argparse.Namespace, prefix: str="") -> dict[str, 'CLIConfig']:
        name_value_pairs = {}
        for attr in fields(cls):
            name = attr.name
            sub_config_cls = attr.type

            if safe_issubclass(sub_config_cls, CLIConfig):
                value = sub_config_cls.from_cli_args(args, prefix)
                name_value_pairs[name] = value
        return name_value_pairs

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser, prefix: str="--") -> argparse.ArgumentParser:
        """
        1. add current config to cli args
        2. add sub configs to cli args
        """
        parser = cls.add_curr_config_cli_args(cls, parser, prefix)
        parser = cls.add_sub_configs_cli_args(cls, parser, prefix)
        return parser

    @staticmethod
    def add_curr_config_cli_args(cls, parser: argparse.ArgumentParser, prefix: str="--") -> argparse.ArgumentParser:
        return parser

    @staticmethod
    def add_sub_configs_cli_args(cls, parser: argparse.ArgumentParser, prefix: str="--") -> argparse.ArgumentParser:
        assert type(cls) == type
        for attr in fields(cls):
            sub_config_cls = attr.type
            if safe_issubclass(sub_config_cls, CLIConfig):
                parser = sub_config_cls.add_cli_args(parser, prefix)
        return parser