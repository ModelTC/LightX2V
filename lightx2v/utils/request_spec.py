"""Request definitions shared by the CLI and runners."""


class Argument:
    def __init__(self, *, cli=True, **options):
        self.cli = cli
        self.options = options

    def add_to(self, parser, name):
        parser.add_argument(f"--{name}", default=None, **self.options)


class RequestSpec:
    def __init__(self, tasks: dict[str, dict[str, Argument]]):
        self.tasks = tasks

    def get_tasks(self, config):
        task = config.get("task", None)
        if task is None:
            raise ValueError("task must be set when the runner is created")
        return (task,)

    def get_arguments(self, config, task):
        return dict(self.tasks[task])

    def get_fields(self, config, task):
        return frozenset(self.get_arguments(config, task))

    def get_cli_arguments(self, config, task):
        return {name: argument for name, argument in self.get_arguments(config, task).items() if argument.cli}

    def prepare_runner(self, runner, args):
        pass


class MultiTaskSpec(RequestSpec):
    def get_tasks(self, config):
        return tuple(self.tasks)
