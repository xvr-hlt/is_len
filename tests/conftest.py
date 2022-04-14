import hydra
import pytest
from is_len import util


@pytest.fixture
def config():
    return util.get_config()


@pytest.fixture
def datamodule(config):
    datamodule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup()
    return datamodule


@pytest.fixture
def experiment(config):
    experiment = hydra.utils.instantiate(config.experiment)
    return experiment
