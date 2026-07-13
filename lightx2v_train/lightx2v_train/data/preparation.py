def prepare_data(config):
    data_config = config.get("data", {})
    data_names = {split_config.get("name") for split_config in data_config.values() if isinstance(split_config, dict)}
    if "libero_fastwam_dataset" in data_names:
        from .libero.preparation import prepare_libero_fastwam_assets

        prepare_libero_fastwam_assets(config)
