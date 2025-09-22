import json
import sys

from loguru import logger


class Pipeline:
    def __init__(self, pipeline_json_file):
        self.pipeline_json_file = pipeline_json_file
        x = json.load(open(pipeline_json_file))
        self.data = x["data"]
        self.meta = x["meta"]
        self.inputs = {}
        self.outputs = {}
        self.temps = {}
        self.model_lists = []
        self.types = {}
        self.queues = set()
        self.model_name_inner_to_outer = self.meta.get("model_name_inner_to_outer", {})
        self.model_name_outer_to_inner = self.meta.get("model_name_outer_to_inner", {})
        self.tidy_pipeline()

    def init_dict(self, base, task, model_cls):
        if task not in base:
            base[task] = {}
        if model_cls not in base[task]:
            base[task][model_cls] = {}

    # tidy each task item eg, ['t2v', 'wan2.1', 'multi_stage']
    def tidy_task(self, task, model_cls, stage, v3):
        out2worker = {}
        out2num = {}
        cur_inps = set()
        cur_temps = set()
        cur_types = {}
        for worker_name, worker_item in v3.items():
            prevs = []
            for inp in worker_item["inputs"]:
                cur_types[inp] = self.get_type(inp)
                if inp in out2worker:
                    prevs.append(out2worker[inp])
                    out2num[inp] -= 1
                    if out2num[inp] <= 0:
                        cur_temps.add(inp)
                else:
                    cur_inps.add(inp)
            worker_item["previous"] = prevs

            for out in worker_item["outputs"]:
                cur_types[out] = self.get_type(out)
                out2worker[out] = worker_name
                if out not in out2num:
                    out2num[out] = 0
                out2num[out] += 1

            if "queue" not in worker_item:
                worker_item["queue"] = "-".join([task, model_cls, stage, worker_name])
            self.queues.add(worker_item["queue"])

        cur_outs = [out for out, num in out2num.items() if num > 0]
        self.inputs[task][model_cls][stage] = list(cur_inps)
        self.outputs[task][model_cls][stage] = cur_outs
        self.temps[task][model_cls][stage] = list(cur_temps)
        self.types[task][model_cls][stage] = cur_types

    # tidy previous dependence workers and queue name
    def tidy_pipeline(self):
        for task, v1 in self.data.items():
            for model_cls, v2 in v1.items():
                for stage, v3 in v2.items():
                    self.init_dict(self.inputs, task, model_cls)
                    self.init_dict(self.outputs, task, model_cls)
                    self.init_dict(self.temps, task, model_cls)
                    self.init_dict(self.types, task, model_cls)
                    self.tidy_task(task, model_cls, stage, v3)
                    self.model_lists.append({"task": task, "model_cls": model_cls, "stage": stage})
        logger.info(f"pipelines: {json.dumps(self.data, indent=4)}")
        logger.info(f"inputs: {self.inputs}")
        logger.info(f"outputs: {self.outputs}")
        logger.info(f"temps: {self.temps}")
        logger.info(f"types: {self.types}")
        logger.info(f"model_lists: {self.model_lists}")
        logger.info(f"queues: {self.queues}")

    def get_item_by_keys(self, keys):
        item = self.data
        for k in keys:
            if k not in item:
                raise Exception(f"{keys} are not in {self.pipeline_json_file}!")
            item = item[k]
        return item

    # eg. keys: ['t2v', 'wan2.1', 'multi_stage', 'text_encoder']
    def get_worker(self, keys):
        return self.get_item_by_keys(keys)

    # eg. keys: ['t2v', 'wan2.1', 'multi_stage']
    def get_workers(self, keys):
        return self.get_item_by_keys(keys)

    # eg. keys: ['t2v', 'wan2.1', 'multi_stage']
    def get_inputs(self, keys):
        item = self.inputs
        for k in keys:
            if k not in item:
                raise Exception(f"{keys} are not in inputs!")
            item = item[k]
        return item

    # eg. keys: ['t2v', 'wan2.1', 'multi_stage']
    def get_outputs(self, keys):
        item = self.outputs
        for k in keys:
            if k not in item:
                raise Exception(f"{keys} are not in outputs!")
            item = item[k]
        return item

    # eg. keys: ['t2v', 'wan2.1', 'multi_stage']
    def get_temps(self, keys):
        item = self.temps
        for k in keys:
            if k not in item:
                raise Exception(f"{keys} are not in temps!")
            item = item[k]
        return item

    # eg. keys: ['t2v', 'wan2.1', 'multi_stage']
    def get_types(self, keys):
        item = self.types
        for k in keys:
            if k not in item:
                raise Exception(f"{keys} are not in types!")
            item = item[k]
        return item

    def check_item_by_keys(self, keys):
        item = self.data
        for k in keys:
            if k not in item:
                return False
            item = item[k]
        return True

    def get_model_lists(self):
        return self.model_lists

    def get_type(self, name):
        return self.meta["special_types"].get(name, "OBJECT")

    def get_monitor_config(self):
        return self.meta["monitor"]

    def get_queues(self):
        return self.queues

    def inner_model_name(self, name):
        return self.model_name_outer_to_inner.get(name, name)

    def outer_model_name(self, name):
        return self.model_name_inner_to_outer.get(name, name)


if __name__ == "__main__":
    pipeline = Pipeline(sys.argv[1])
    print(pipeline.get_workers(["t2v", "wan2.1", "multi_stage"]))
    print(pipeline.get_worker(["i2v", "wan2.1", "multi_stage", "dit"]))
