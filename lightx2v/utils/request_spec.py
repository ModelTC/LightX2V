"""Request definitions shared by the CLI, HTTP API and runners."""

import argparse
import json
from typing import Literal, get_args, get_origin

OutputType = Literal["image", "video", "text", "mesh", "depth", "reconstruction"]


class Argument:
    def __init__(self, type=str, *, cli=True, server=True, gt=None, ge=None, lt=None, le=None, strict=None, **options):
        self.type = type
        self.cli = cli
        self.server = server
        self.options = options
        self.constraints = {name: value for name, value in dict(gt=gt, ge=ge, lt=lt, le=le, strict=strict).items() if value is not None}

    def add_to(self, parser, name):
        options = dict(self.options)
        if self.type is bool:
            options.setdefault("action", argparse.BooleanOptionalAction)
        elif get_origin(self.type) is list:
            options.update(type=get_args(self.type)[0], nargs="+")
        elif self.type in (str, int, float):
            options["type"] = self.type
        else:
            options["type"] = self._parse_cli_value
        parser.add_argument(f"--{name}", default=None, **options)

    def _parse_cli_value(self, value):
        if str in get_args(self.type) and not value.lstrip().startswith(("[", "{")):
            return value
        return json.loads(value)

    def model_field(self):
        from pydantic import Field

        annotation = Literal[tuple(self.options["choices"])] if "choices" in self.options else self.type
        return annotation | None, Field(default=None, description=self.options.get("help"), **self.constraints)


def create_request_model(name, arguments, *, base=None, validators=None, **fields):
    from pydantic import ConfigDict, create_model

    options = {"__base__": base} if base is not None else {"__config__": ConfigDict(extra="forbid")}
    if validators:
        options["__validators__"] = validators
    definitions = {name: argument.model_field() for name, argument in arguments.items()}
    return create_model(name, **options, **(definitions | fields))


class TaskSpec:
    def __init__(self, arguments: dict[str, Argument], *, outputs: tuple[OutputType, ...]):
        self.arguments = arguments
        self.outputs = outputs

    def get_output_type(self, inputs):
        (output,) = self.outputs
        return output


class RequestSpec:
    def __init__(self, tasks: dict[str, TaskSpec]):
        self.tasks = tasks

    def get_tasks(self, config):
        task = config.get("task", None)
        if task is None:
            raise ValueError("task must be set when the runner is created")
        return (task,)

    def get_arguments(self, config, task):
        return dict(self.tasks[task].arguments)

    def get_tasks_for_output(self, config, output: OutputType):
        return tuple(task for task in self.get_tasks(config) if output in self.tasks[task].outputs)

    def get_fields(self, config, task):
        return frozenset(self.get_arguments(config, task))

    def get_cli_arguments(self, config, task):
        return {name: argument for name, argument in self.get_arguments(config, task).items() if argument.cli}

    def prepare_runner(self, runner, args):
        pass


class MultiTaskSpec(RequestSpec):
    def get_tasks(self, config):
        return tuple(self.tasks)
