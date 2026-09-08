def get_offload_plan(config):
    plan = config.get("offload_plan")
    if plan is not None:
        return plan

    return {
        "offload_granularity": config.get("offload_granularity", "block"),
        "use_event_offload": config.get("use_event_offload", False),
        "resident_blocks": {},
    }


def get_offload_granularity(config):
    return get_offload_plan(config).get("offload_granularity", "block")


def use_event_offload(config):
    return get_offload_plan(config).get("use_event_offload", False)


def normalize_offload_plan(config):
    if "offload_plan" in config:
        plan = dict(config["offload_plan"])
    else:
        plan = get_offload_plan(config)
    plan.setdefault("offload_granularity", "block")
    plan.setdefault("use_event_offload", False)
    plan.setdefault("resident_blocks", {})
    config["offload_plan"] = plan
