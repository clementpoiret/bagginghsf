import numpy as np
import torch


def squash(x,
           squashing_constant=1.,
           dim=-1,
           eps=1e-7,
           safe=True,
           p=2,
           **kwargs):
    if safe:
        squared_norm = torch.sum(torch.square(x), axis=dim, keepdim=True)
        safe_norm = torch.sqrt(squared_norm + eps)
        squash_factor = squared_norm / (squashing_constant + squared_norm)
        unit_vector = x / safe_norm

        return squash_factor * unit_vector
    else:
        norm = x.norm(dim=dim, keepdim=True, p=p)
        norm_squared = norm * norm
        return (x / norm) * (norm_squared / (squashing_constant + norm_squared))


def smsquash(x, caps_dim=1, atoms_dim=2, eps=1e-7, **kwargs):
    """Softmax Squash (Poiret, et al., 2021)
    Novel squash function. It rescales caps 2-norms to a probability
    distribution over predicted output classes."""
    squared_norm = torch.sum(torch.square(x), axis=atoms_dim, keepdim=True)
    safe_norm = torch.sqrt(squared_norm + eps)

    a = torch.exp(safe_norm) / torch.sum(torch.exp(safe_norm), axis=caps_dim)
    b = x / safe_norm

    return a * b


def safe_length(x, dim=2, keepdim=False, eps=1e-7):
    squared_norm = torch.sum(torch.square(x), axis=dim, keepdim=keepdim)
    return torch.sqrt(squared_norm + eps)


def calc_same_padding(
    input_,
    kernel=1,
    stride=1,
    dilation=1,
    transposed=False,
):
    if transposed:
        return (dilation * (kernel - 1) + 1) // 2 - 1, input_ // (1. / stride)
    else:
        return (dilation * (kernel - 1) + 1) // 2, input_ // stride


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2**k for k in range(num_levels)]


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    assert input.dim() == 4

    # expand the input tensor to Nx1xDxHxW before scattering
    input = input.unsqueeze(1)
    # create result tensor shape (NxCxDxHxW)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)
