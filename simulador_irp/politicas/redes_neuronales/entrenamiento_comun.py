import copy
from typing import List, Sequence

import numpy as np
import torch

from simulador_irp.nucleo.estado import Estado
from simulador_irp.politicas.aproximacion_funcion_valor import AproximadorFuncionValor
from simulador_irp.politicas.redes_neuronales.red_valor import RedValor


class AproximadorRedNeuronal(AproximadorFuncionValor):
    """Aproximador de Q(s, a) con una red neuronal; conserva una copia de la mejor red encontrada."""

    def __init__(
        self,
        instancia,
        simulador,
        episodios: int,
        epsilon: float,
        tasa_aprendizaje: float,
        neuronas_por_capa_oculta: Sequence[int],
    ):
        super().__init__(instancia, simulador, episodios, epsilon, tasa_aprendizaje)
        self.dimension_entrada = self.cantidad_caracteristicas()
        self.red = RedValor(self.dimension_entrada, neuronas_por_capa_oculta)
        self.optimizador = torch.optim.Adam(self.red.parameters(), lr=self.tasa_aprendizaje)
        self.funcion_perdida = torch.nn.MSELoss()
        self.mejor_red = None

    def seleccionar_mejor_accion(self, estado: Estado, acciones: List[dict]) -> dict:
        """Evalúa todas las acciones en un solo lote y devuelve la de menor valor Q estimado."""
        self.red.eval()
        with torch.no_grad():
            caracteristicas = [self.caracteristicas(estado, accion) for accion in acciones]
            lote = torch.tensor(np.array(caracteristicas), dtype=torch.float32)
            valores = self.red(lote).squeeze(-1).numpy()
        return acciones[np.argmin(valores)]

    def _guardar_mejor_modelo(self) -> None:
        self.mejor_red = copy.deepcopy(self.red)
