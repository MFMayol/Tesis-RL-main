import numpy as np

from simulador_irp.nucleo.estado import Estado
from simulador_irp.politicas.heuristicas import (
    PoliticaHeuristicaBase,
    PoliticaHeuristicaClusterizada,
    PoliticaHeuristicaSimple,
)
from simulador_irp.politicas.politica_base import PoliticaBase

SIMULACIONES_POR_ACCION = 5


class PoliticaRollout(PoliticaHeuristicaBase):
    """Elige la acción cuyo costo futuro esperado es menor, estimado simulando una política heurística base.

    Para cada acción candidata se simulan varios episodios aplicándola y siguiendo después la política base.
    """

    def __init__(self, instancia, simulador, politica_base: PoliticaBase):
        super().__init__(instancia, simulador)
        self.politica_base = politica_base

    def tomar_accion(self, estado: Estado) -> dict:
        costo_esperado_por_accion = []
        for accion in self.obtener_acciones(estado):
            costos = np.array([self._simular_costo(accion, estado) for _ in range(SIMULACIONES_POR_ACCION)])
            costo_esperado_por_accion.append((accion, np.mean(costos)))
        return min(costo_esperado_por_accion, key=lambda par: par[1])[0]

    def _simular_costo(self, accion: dict, estado: Estado) -> float:
        """Costo de aplicar la acción y continuar con la política base hasta el horizonte."""
        transicion = self.simulador.transicion(estado, accion)
        estado = transicion.estado_nuevo
        costo_total = transicion.costo_total
        while True:
            accion = self.politica_base.tomar_accion(estado)
            transicion = self.simulador.transicion(estado, accion)
            costo_total += transicion.costo_total
            if self.es_terminal(transicion.estado_nuevo):
                return costo_total
            estado = transicion.estado_nuevo


class PoliticaRolloutHeuristicaSimple(PoliticaRollout):
    """Rollout con la política heurística simple como política base."""

    def __init__(self, instancia, simulador):
        super().__init__(instancia, simulador, PoliticaHeuristicaSimple(instancia, simulador))


class PoliticaRolloutHeuristicaClusterizada(PoliticaRollout):
    """Rollout con la política heurística clusterizada como política base."""

    def __init__(self, instancia, simulador):
        super().__init__(instancia, simulador, PoliticaHeuristicaClusterizada(instancia, simulador))
