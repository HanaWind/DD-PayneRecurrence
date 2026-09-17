from ddpayne.data.build import stable_split


def test_stable_split_keeps_repeated_source_together():
    config = {
        "train_fraction": 0.8,
        "validation_fraction": 0.1,
        "test_fraction": 0.1,
        "seed": 42,
    }
    assert stable_split("Gaia-123", config) == stable_split("Gaia-123", config)


def test_stable_split_returns_known_code():
    config = {
        "train_fraction": 0.8,
        "validation_fraction": 0.1,
        "test_fraction": 0.1,
        "seed": 42,
    }
    assert stable_split("any-source", config) in {0, 1, 2}
