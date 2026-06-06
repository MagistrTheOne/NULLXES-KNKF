from pathlib import Path

from knk.model.param_counter import count_parameters, load_config, validate_targets


def test_target_param_config_validates():
    config = load_config(Path("configs/model/knk_vf_target_70b_active.yaml"))
    report = count_parameters(config)
    assert validate_targets(config, report) == []


def test_proxy_param_count_is_positive():
    config = load_config(Path("configs/model/knk_vf_proxy_3b_active.yaml"))
    report = count_parameters(config)
    assert report.total_params > report.active_params > 0
