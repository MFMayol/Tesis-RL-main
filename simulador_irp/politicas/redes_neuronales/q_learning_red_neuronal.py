import copy
import random
import time
from collections import deque
from typing import NamedTuple, Sequence

import numpy as np
import torch

from simulador_irp.hiperparametros import EXPERIENCIAS_MINIMAS_POR_LOTE
from simulador_irp.nucleo.estado import Estado
from simulador_irp.politicas.aproximacion_funcion_valor import EPISODIOS_EVALUACION, PERIODO_EVALUACION_EPISODIOS
from simulador_irp.politicas.entregas_parciales import EntregasParcialesMixin
from simulador_irp.politicas.redes_neuronales.entrenamiento_comun import AproximadorRedNeuronal


class Experiencia(NamedTuple):
    estado: Estado
    accion: dict
    caracteristicas: np.ndarray
    recompensa: float
    estado_siguiente: Estado
    es_final: bool


class QLearningRedNeuronal(AproximadorRedNeuronal):
    """Q-learning profundo (Double DQN) con memoria de experiencias, red objetivo y actualización suave."""

    def __init__(
        self,
        instancia,
        simulador,
        episodios: int,
        epsilon: float,
        tasa_aprendizaje: float,
        factor_descuento: float,
        tamano_lote: int,
        capacidad_memoria_experiencias: int,
        neuronas_por_capa_oculta: Sequence[int],
    ):
        super().__init__(instancia, simulador, episodios, epsilon, tasa_aprendizaje, neuronas_por_capa_oculta)
        self.factor_descuento = factor_descuento
        self.tamano_lote = tamano_lote
        self.memoria_experiencias = deque(maxlen=capacidad_memoria_experiencias)
        self.experiencias_minimas_para_entrenar = tamano_lote * EXPERIENCIAS_MINIMAS_POR_LOTE  # Espera diversidad antes de aprender.
        self.red_objetivo = copy.deepcopy(self.red)
        self.red_objetivo.eval()
        self.tau = 0.005  # Factor de la actualización suave (Polyak) de la red objetivo.
        self.pasos_totales = 0
        self.pasos_entre_actualizaciones = 4

    def entrenar_modelo(self) -> None:
        self.actualizar_mejor_modelo(EPISODIOS_EVALUACION)  # Punto de partida antes de entrenar.

        inicio = time.perf_counter()
        for episodio in range(self.episodios):
            estado = self.simulador.determinar_estado_inicial()
            while not self.es_terminal(estado):
                accion = self._elegir_accion_epsilon_greedy(estado)
                caracteristicas = self.caracteristicas(estado, accion)
                transicion = self.simulador.transicion(estado, accion)
                es_final = self.es_terminal(transicion.estado_nuevo)

                self.memoria_experiencias.append(
                    Experiencia(estado, accion, caracteristicas, transicion.costo_total, transicion.estado_nuevo, es_final)
                )
                estado = transicion.estado_nuevo
                self.pasos_totales += 1

                hay_experiencias_suficientes = len(self.memoria_experiencias) >= self.experiencias_minimas_para_entrenar
                if hay_experiencias_suficientes and self.pasos_totales % self.pasos_entre_actualizaciones == 0:
                    self._aprender_de_experiencias()

            if episodio % PERIODO_EVALUACION_EPISODIOS == 0:
                self.actualizar_mejor_modelo()
                print(f"Episodio {episodio:5d} | mejor FO: {self.mejor_funcion_objetivo:10.2f}")

        self.tiempo_entrenamiento_segundos = time.perf_counter() - inicio

    def _aprender_de_experiencias(self) -> None:
        """Un paso de gradiente sobre un lote aleatorio de experiencias."""
        if len(self.memoria_experiencias) < self.tamano_lote:
            return

        self.red.train()
        lote = random.sample(self.memoria_experiencias, self.tamano_lote)
        caracteristicas, recompensas, estados_siguientes, es_final = (
            [experiencia.caracteristicas for experiencia in lote],
            [experiencia.recompensa for experiencia in lote],
            [experiencia.estado_siguiente for experiencia in lote],
            [experiencia.es_final for experiencia in lote],
        )

        caracteristicas_tensor = torch.tensor(np.array(caracteristicas), dtype=torch.float32)
        recompensas_tensor = torch.tensor(recompensas, dtype=torch.float32)
        es_final_tensor = torch.tensor(es_final, dtype=torch.bool)

        no_es_final = ~es_final_tensor
        estados_no_finales = [estado for indice, estado in enumerate(estados_siguientes) if no_es_final[indice]]

        valores_siguientes = torch.zeros(self.tamano_lote, dtype=torch.float32)
        if estados_no_finales:
            caracteristicas_mejores_acciones = []
            with torch.no_grad():
                for estado_siguiente in estados_no_finales:
                    acciones_siguientes = self.obtener_acciones(estado_siguiente)
                    if not acciones_siguientes:
                        caracteristicas_mejores_acciones.append(np.zeros(self.dimension_entrada, dtype=np.float32))
                        continue
                    # Double DQN: la red principal elige la acción y la red objetivo la evalúa.
                    mejor_accion = self.seleccionar_mejor_accion(estado_siguiente, acciones_siguientes)
                    caracteristicas_mejores_acciones.append(self.caracteristicas(estado_siguiente, mejor_accion))

            lote_mejores_acciones = torch.tensor(np.array(caracteristicas_mejores_acciones), dtype=torch.float32)
            with torch.no_grad():
                valores_siguientes[no_es_final] = self.red_objetivo(lote_mejores_acciones).squeeze(-1)

        objetivos = recompensas_tensor + self.factor_descuento * valores_siguientes

        self.red.train()  # seleccionar_mejor_accion deja la red en modo evaluación.
        valores_actuales = self.red(caracteristicas_tensor)
        perdida = self.funcion_perdida(valores_actuales, objetivos.unsqueeze(1))

        self.optimizador.zero_grad()
        perdida.backward()
        self.optimizador.step()

        for parametro_objetivo, parametro in zip(self.red_objetivo.parameters(), self.red.parameters()):
            parametro_objetivo.data.copy_(self.tau * parametro.data + (1.0 - self.tau) * parametro_objetivo.data)


class QLearningRedNeuronalEntregasParciales(EntregasParcialesMixin, QLearningRedNeuronal):
    """Q-learning con red neuronal que además evalúa entregas parciales (100 %, 75 %, 50 % y 25 %)."""
