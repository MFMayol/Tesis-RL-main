import time
from typing import List

import numpy as np
import torch

from simulador_irp.politicas.aproximacion_funcion_valor import EPISODIOS_EVALUACION, PERIODO_EVALUACION_EPISODIOS
from simulador_irp.politicas.entregas_parciales import EntregasParcialesMixin
from simulador_irp.politicas.redes_neuronales.entrenamiento_comun import AproximadorRedNeuronal


class MonteCarloRedNeuronal(AproximadorRedNeuronal):
    """Monte Carlo on-policy con red neuronal.

    Cada episodio completo produce los retornos observados y se da un único paso de gradiente sobre el episodio
    entero. Al terminar se restaura la mejor red evaluada.
    """

    def entrenar_modelo(self) -> None:
        self.actualizar_mejor_modelo(EPISODIOS_EVALUACION)  # Punto de partida antes de entrenar.

        inicio = time.perf_counter()
        for episodio in range(self.episodios):
            pasos = self.ejecutar_episodio_epsilon_greedy()

            retorno = 0.0
            caracteristicas_episodio: List[np.ndarray] = []
            retornos_episodio: List[List[float]] = []
            for paso in reversed(pasos):
                retorno = paso.recompensa + retorno
                caracteristicas_episodio.append(paso.caracteristicas)
                retornos_episodio.append([retorno])

            entradas = torch.tensor(np.array(caracteristicas_episodio), dtype=torch.float32)
            objetivos = torch.tensor(retornos_episodio, dtype=torch.float32)

            self.red.train()
            self.optimizador.zero_grad()
            perdida = self.funcion_perdida(self.red(entradas), objetivos)
            perdida.backward()
            self.optimizador.step()

            if episodio % PERIODO_EVALUACION_EPISODIOS == 0:
                self.actualizar_mejor_modelo()
                print(
                    f"Episodio {episodio:5d} | pérdida: {perdida.item():10.2f} | "
                    f"FO actual: {self.registro_funcion_objetivo[-1]:10.2f} | mejor FO: {self.mejor_funcion_objetivo:10.2f}"
                )

        self.tiempo_entrenamiento_segundos = time.perf_counter() - inicio
        self.red = self.mejor_red  # El método es inestable: se conserva la mejor red encontrada.


class MonteCarloRedNeuronalEntregasParciales(EntregasParcialesMixin, MonteCarloRedNeuronal):
    """Monte Carlo con red neuronal que además evalúa entregas parciales (100 %, 75 %, 50 % y 25 %)."""
