"""Build the inference CLI from the selected model, task and startup config."""

import argparse
import sys

from lightx2v.models.runners.registry import RUNNER_MODULES, get_model_spec
from lightx2v.utils.set_config import build_startup_config


def _startup_parser(*, add_help=True, required=True):
    parser = argparse.ArgumentParser(add_help=add_help, allow_abbrev=False)
    parser.add_argument("--model_cls", required=required, choices=RUNNER_MODULES, help="Model implementation.")
    parser.add_argument("--task", help="Inference task.")
    parser.add_argument("--model_path", required=required, help="Model checkpoint directory.")
    parser.add_argument("--config_json", required=required, help="Startup configuration file.")
    parser.add_argument("--model-variant", help="Model-specific startup weight variant; overrides the configuration file.")
    return parser


def build_parser(config=None, task=None):
    parser = _startup_parser()
    if config is None:
        parser.description = "Add --model_cls, --task, --model_path and --config_json to see the accepted inputs."
        return parser

    spec = get_model_spec(config["model_cls"])
    parser.description = f"Inputs for {config['model_cls']} / {task}. Omitted inputs use the model defaults."
    group = parser.add_argument_group("request inputs")
    for name, argument in spec.get_cli_arguments(config, task).items():
        argument.add_to(group, name)
    return parser


def parse_cli_inputs(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    startup, _ = _startup_parser(add_help=False, required=False).parse_known_args(argv)
    parser = build_parser()
    has_config = all((startup.model_cls, startup.model_path, startup.config_json))
    wants_help = "-h" in argv or "--help" in argv
    if not has_config or (wants_help and startup.task is None):
        if startup.model_cls:
            tasks = ", ".join(get_model_spec(startup.model_cls).tasks)
            parser.description += f" Supported tasks for {startup.model_cls}: {tasks}."
        parser.parse_args(argv)

    try:
        config = build_startup_config({key: value for key, value in vars(startup).items() if value is not None}, configure_runtime=False)
        spec = get_model_spec(config["model_cls"])
        if startup.task is not None and startup.task not in spec.tasks:
            parser.error(f"{config['model_cls']} does not support task {startup.task!r}; choose from: {', '.join(spec.tasks)}")
        tasks = spec.get_tasks(config)
        task = startup.task
        if task is None:
            if len(tasks) != 1:
                parser.error(f"--task is required; choose from: {', '.join(tasks)}")
            task = tasks[0]
        if task not in tasks:
            parser.error(f"{config['model_cls']} does not support task {task!r} with this configuration; choose from: {', '.join(tasks)}")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    args = build_parser(config, task).parse_args(argv)
    request_fields = spec.get_fields(config, task)
    request = {key: value for key, value in vars(args).items() if key in request_fields and value is not None}
    request["task"] = task
    return args, config, request
