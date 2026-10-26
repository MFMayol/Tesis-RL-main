from typing import Sequence

import torch
import torch.nn as nn


class RedValor(nn.Module):
    """Perceptrón multicapa que aproxima el valor Q(s, a): entrada = características, salida = una neurona."""

    def __init__(self, dimension_entrada: int, neuronas_por_capa_oculta: Sequence[int]):
        super().__init__()
        capas = []
        dimension_actual = dimension_entrada
        for neuronas in neuronas_por_capa_oculta:
            capas.append(nn.Linear(dimension_actual, neuronas))
            capas.append(nn.ReLU())
            dimension_actual = neuronas
        capas.append(nn.Linear(dimension_actual, 1))
        self.modelo = nn.Sequential(*capas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.modelo(x)
