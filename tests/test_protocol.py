from pathlib import Path

from iot_benchmark.protocol import load_protocol, protocol_sha256, validate_protocol, write_lock


def test_protocol_has_three_benchmarks_seven_method_families_and_five_seeds(tmp_path: Path):
    protocol = load_protocol()
    assert len(protocol["benchmarks"]) == 3
    assert len(protocol["methods"]["required_external_families"]) == 7
    assert len(protocol["random_seeds"]) == 5
    validate_protocol(protocol)
    assert len(protocol_sha256()) == 64
    lock = write_lock(tmp_path / "lock.json")
    assert lock.exists()


def test_prospective_mode_forbids_future_information():
    protocol = load_protocol()
    prospective = protocol["information_modes"]["prospective_prediction"]
    assert prospective["target_observations_allowed"] is False
    assert prospective["target_marginal_allowed"] is False
    assert prospective["future_clone_graph_allowed_when_method_contract_requires"] is False
