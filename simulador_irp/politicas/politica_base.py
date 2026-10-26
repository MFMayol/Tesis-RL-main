from abc import ABC, abstractmethod
from typing import Iterable, NamedTuple

import numpy as np

from simulador_irp.nucleo.estado import Estado
from simulador_irp.nucleo.instancia import Instancia
from simulador_irp.nucleo.simulador_transicion import SimuladorTransicion


class ResultadoSimulacion(NamedTuple):
    costo_total: float
    costo_traslado: float
    costo_demanda_insatisfecha: float
    costo_almacenamiento: float
    periodos: float


def promediar_resultados(resultados: Iterable[ResultadoSimulacion]) -> ResultadoSimulacion:
    """Promedia campo a campo una colección de resultados de simulación."""
    resultados = list(resultados)
    return ResultadoSimulacion(*(np.mean([resultado[campo] for resultado in resultados]) for campo in range(5)))


class PoliticaBase(ABC):
    """Política de decisión que se evalúa simulando un episodio completo."""

    def __init__(self, instancia: Instancia, simulador: SimuladorTransicion):
        self.instancia = instancia
        self.simulador = simulador

    @abstractmethod
    def tomar_accion(self, estado: Estado) -> dict:
        """Devuelve la acción {id_vehiculo: {id_destino: {id_producto: cantidad}}} para el estado dado."""

    def simular_episodio(self) -> ResultadoSimulacion:
        """Simula un episodio desde el estado inicial hasta el horizonte aplicando esta política."""
        estado = self.simulador.determinar_estado_inicial()
        costo_total = 0
        costo_traslado = 0
        costo_demanda_insatisfecha = 0
        costo_almacenamiento = 0
        periodos = 0
        while True:
            accion = self.tomar_accion(estado)
            transicion = self.simulador.transicion(estado, accion)
            costo_total += transicion.costo_total
            costo_traslado += transicion.costo_traslado
            costo_demanda_insatisfecha += transicion.costo_demanda_insatisfecha
            costo_almacenamiento += transicion.costo_almacenamiento
            periodos += 1
            if self.es_terminal(transicion.estado_nuevo):
                break
            estado = transicion.estado_nuevo
        return ResultadoSimulacion(costo_total, costo_traslado, costo_demanda_insatisfecha, costo_almacenamiento, periodos)

    def es_terminal(self, estado: Estado) -> bool:
        return estado.tiempo >= self.instancia.horizonte_tiempo

    def planificar_entrega(self, inventario_vehiculo: dict, demanda_media_cliente: dict) -> dict:
        """Reparte el inventario del vehículo entre los productos en proporción a la demanda media del cliente."""
        demanda_total = sum(demanda_media_cliente.values())
        inventario_total = sum(inventario_vehiculo.values())

        if not demanda_total or not inventario_total:
            return dict.fromkeys(demanda_media_cliente, 0)

        proporcion = inventario_total / demanda_total
        return {
            id_producto: min(round(demanda * proporcion), inventario_vehiculo.get(id_producto, 0))
            for id_producto, demanda in demanda_media_cliente.items()
        }
