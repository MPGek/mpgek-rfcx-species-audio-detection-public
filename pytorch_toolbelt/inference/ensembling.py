from torch import nn, Tensor
from typing import List, Union

__all__ = ["ApplySoftmaxTo", "ApplySigmoidTo", "Ensembler", "PickModelOutput"]


class ApplySoftmaxTo(nn.Module):
    def __init__(self, model: nn.Module, output_key: Union[str, List[str]] = "logits", dim=1, temperature=1):
        """
        Apply softmax activation on given output(s) of the model
        :param model: Model to wrap
        :param output_key: string or list of strings, indicating to what outputs softmax activation should be applied.
        :param dim: Tensor dimension for softmax activation
        :param temperature: Temperature scaling coefficient. Values > 1 will make logits sharper.
        """
        super().__init__()
        output_key = output_key if isinstance(output_key, (list, tuple)) else [output_key]
        # By converting to set, we prevent double-activation by passing output_key=["logits", "logits"]
        self.output_keys = set(output_key)
        self.model = model
        self.dim = dim
        self.temperature = temperature

    def forward(self, *input, **kwargs):
        output = self.model(*input, **kwargs)
        for key in self.output_keys:
            output[key] = output[key].mul(self.temperature).softmax(dim=1)
        return output


class ApplySigmoidTo(nn.Module):
    def __init__(self, model: nn.Module, output_key: Union[str, List[str]] = "logits", temperature=1):
        """
        Apply sigmoid activation on given output(s) of the model
        :param model: Model to wrap
        :param output_key: string or list of strings, indicating to what outputs sigmoid activation should be applied.
        :param temperature: Temperature scaling coefficient. Values > 1 will make logits sharper.
        """
        super().__init__()
        output_key = output_key if isinstance(output_key, (list, tuple)) else [output_key]
        # By converting to set, we prevent double-activation by passing output_key=["logits", "logits"]
        self.output_keys = set(output_key)
        self.model = model
        self.temperature = temperature

    def forward(self, *input, **kwargs):  # skipcq: PYL-W0221
        output = self.model(*input, **kwargs)
        for key in self.output_keys:
            output[key] = output[key].mul(self.temperature).sigmoid()
        return output


class Ensembler(nn.Module):
    """
    Compute sum (or average) of outputs of several models.
    """

    def __init__(self, models: List[nn.Module], average=True, outputs=None):
        """

        :param models:
        :param average:
        :param outputs: Name of model outputs to average and return from Ensembler.
            If None, all outputs from the first model will be used.
        """
        super().__init__()
        self.outputs = outputs
        self.models = nn.ModuleList(models)
        self.average = average

    def forward(self, *input, **kwargs):  # skipcq: PYL-W0221
        output_0 = self.models[0](*input, **kwargs)
        num_models = len(self.models)

        if self.outputs:
            keys = self.outputs
        else:
            keys = output_0.keys()

        for index in range(1, num_models):
            output_i = self.models[index](*input, **kwargs)

            # Sum outputs
            for key in keys:
                output_0[key].add_(output_i[key])

        if self.average:
            for key in keys:
                output_0[key].mul_(1.0 / num_models)

        return output_0


class PickModelOutput(nn.Module):
    """
    Assuming you have a model that outputs a dictionary, this module returns only a given element by it's key
    """

    def __init__(self, model: nn.Module, key: str):
        super().__init__()
        self.model = model
        self.target_key = key

    def forward(self, *input, **kwargs) -> Tensor:
        output = self.model(*input, **kwargs)
        return output[self.target_key]
