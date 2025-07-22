import torch


def cache_init(num_steps, model_kwargs=None):
    """
    Initialization for cache.
    """
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1] = {}
    cache_index[-1] = {}
    cache_index["layer_index"] = {}
    cache_dic["attn_map"] = {}
    cache_dic["attn_map"][-1] = {}
    cache_dic["attn_map"][-1]["double_stream"] = {}
    cache_dic["attn_map"][-1]["single_stream"] = {}

    cache_dic["k-norm"] = {}
    cache_dic["k-norm"][-1] = {}
    cache_dic["k-norm"][-1]["double_stream"] = {}
    cache_dic["k-norm"][-1]["single_stream"] = {}

    cache_dic["v-norm"] = {}
    cache_dic["v-norm"][-1] = {}
    cache_dic["v-norm"][-1]["double_stream"] = {}
    cache_dic["v-norm"][-1]["single_stream"] = {}

    cache_dic["cross_attn_map"] = {}
    cache_dic["cross_attn_map"][-1] = {}
    cache[-1]["double_stream"] = {}
    cache[-1]["single_stream"] = {}
    cache_dic["cache_counter"] = 0

    for j in range(20):
        cache[-1]["double_stream"][j] = {}
        cache_index[-1][j] = {}
        cache_dic["attn_map"][-1]["double_stream"][j] = {}
        cache_dic["attn_map"][-1]["double_stream"][j]["total"] = {}
        cache_dic["attn_map"][-1]["double_stream"][j]["txt_mlp"] = {}
        cache_dic["attn_map"][-1]["double_stream"][j]["img_mlp"] = {}

        cache_dic["k-norm"][-1]["double_stream"][j] = {}
        cache_dic["k-norm"][-1]["double_stream"][j]["txt_mlp"] = {}
        cache_dic["k-norm"][-1]["double_stream"][j]["img_mlp"] = {}

        cache_dic["v-norm"][-1]["double_stream"][j] = {}
        cache_dic["v-norm"][-1]["double_stream"][j]["txt_mlp"] = {}
        cache_dic["v-norm"][-1]["double_stream"][j]["img_mlp"] = {}

    for j in range(40):
        cache[-1]["single_stream"][j] = {}
        cache_index[-1][j] = {}
        cache_dic["attn_map"][-1]["single_stream"][j] = {}
        cache_dic["attn_map"][-1]["single_stream"][j]["total"] = {}

        cache_dic["k-norm"][-1]["single_stream"][j] = {}
        cache_dic["k-norm"][-1]["single_stream"][j]["total"] = {}

        cache_dic["v-norm"][-1]["single_stream"][j] = {}
        cache_dic["v-norm"][-1]["single_stream"][j]["total"] = {}

    cache_dic["taylor_cache"] = False
    cache_dic["duca"] = False
    cache_dic["test_FLOPs"] = False

    mode = "Taylor"
    if mode == "original":
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.0
        cache_dic["fresh_threshold"] = 1
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = 1

    elif mode == "ToCa":
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.10
        cache_dic["fresh_threshold"] = 5
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = 1
        cache_dic["duca"] = False

    elif mode == "DuCa":
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.10
        cache_dic["fresh_threshold"] = 5
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = 1
        cache_dic["duca"] = True

    elif mode == "Taylor":
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.0
        cache_dic["fresh_threshold"] = 2
        cache_dic["max_order"] = 1
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["taylor_cache"] = True
        cache_dic["first_enhance"] = 1

    current = {}
    current["num_steps"] = num_steps
    current["activated_steps"] = [0]

    return cache_dic, current


def force_scheduler(cache_dic, current):
    if cache_dic["fresh_ratio"] == 0:
        # FORA
        linear_step_weight = 0.0
    else:
        # TokenCache
        linear_step_weight = 0.0
    step_factor = torch.tensor(1 - linear_step_weight + 2 * linear_step_weight * current["step"] / current["num_steps"])
    threshold = torch.round(cache_dic["fresh_threshold"] / step_factor)

    # no force constrain for sensitive steps, cause the performance is good enough.
    # you may have a try.

    cache_dic["cal_threshold"] = threshold
    # return threshold


def cal_type(cache_dic, current):
    """
    Determine calculation type for this step
    """
    if (cache_dic["fresh_ratio"] == 0.0) and (not cache_dic["taylor_cache"]):
        # FORA:Uniform
        first_step = current["step"] == 0
    else:
        # ToCa: First enhanced
        first_step = current["step"] < cache_dic["first_enhance"]
        # first_step = (current['step'] <= 3)

    force_fresh = cache_dic["force_fresh"]
    if not first_step:
        fresh_interval = cache_dic["cal_threshold"]
    else:
        fresh_interval = cache_dic["fresh_threshold"]

    if (first_step) or (cache_dic["cache_counter"] == fresh_interval - 1):
        current["type"] = "full"
        cache_dic["cache_counter"] = 0
        current["activated_steps"].append(current["step"])
        # current['activated_times'].append(current['t'])
        force_scheduler(cache_dic, current)

    elif cache_dic["taylor_cache"]:
        cache_dic["cache_counter"] += 1
        current["type"] = "taylor_cache"

    else:
        cache_dic["cache_counter"] += 1
        if cache_dic["duca"]:
            if cache_dic["cache_counter"] % 2 == 1:  # 0: ToCa-Aggresive-ToCa, 1: Aggresive-ToCa-Aggresive
                current["type"] = "ToCa"
            # 'cache_noise' 'ToCa' 'FORA'
            else:
                current["type"] = "aggressive"
        else:
            current["type"] = "ToCa"

        # if current['step'] < 25:
        #    current['type'] = 'FORA'
        # else:
        #    current['type'] = 'aggressive'
