import pytest
import torch

from model import SketchCNN


@pytest.fixture
def model():
    return SketchCNN(num_classes=12)


def test_model_instantiates_with_requested_class_count(model):
    assert model.classifier[-1].out_features == 12


@pytest.mark.parametrize("batch_size", [1, 4])
def test_forward_returns_one_logit_vector_per_input(model, batch_size):
    outputs = model(torch.zeros(batch_size, 1, 28, 28))

    assert outputs.shape == (batch_size, 12)


def test_forward_rejects_inputs_with_wrong_channel_count(model):
    with pytest.raises(RuntimeError, match="expected input.*to have 1 channels"):
        model(torch.zeros(2, 3, 28, 28))


def test_forward_rejects_inputs_with_wrong_spatial_size(model):
    with pytest.raises(RuntimeError, match="mat1 and mat2 shapes cannot be multiplied"):
        model(torch.zeros(2, 1, 32, 32))
