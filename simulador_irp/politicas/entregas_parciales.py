import math
from typing import Dict, List, Set, Tuple

from simulador_irp.politicas.aproximacion_funcion_valor import MonteCarloFourier


class EntregasParcialesMixin:
    """Ofrece, además de la entrega completa, entregas parciales del plan (por ejemplo 75 %, 50 % y 25 %)."""

    factores_entrega: Tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)

    def _planes_de_entrega(self, plan_completo: dict, cliente) -> List[dict]:
        if not plan_completo or sum(plan_completo.values()) <= 0:
            return []

        planes = []
        planes_vistos = set()
        for factor in self.factores_entrega:
            plan_parcial = {
                id_producto: round(cantidad * factor)
                for id_producto, cantidad in plan_completo.items()
                if round(cantidad * factor) > 0
            }
            if not plan_parcial:
                continue
            identificador = tuple(sorted(plan_parcial.items()))
            if identificador not in planes_vistos:
                planes_vistos.add(identificador)
                planes.append(plan_parcial)
        return planes


class MonteCarloFourierEntregasParciales(EntregasParcialesMixin, MonteCarloFourier):
    """Monte Carlo con base de Fourier y entregas parciales; útil cuando almacenar tiene costo."""

    factores_entrega = (1.0, 0.5, 0.75, 0.25)


class MonteCarloFourierCriticidad(MonteCarloFourier):
    """Monte Carlo con base de Fourier que ofrece entregas parciales filtradas por criticidad del producto.

    La criticidad de un producto es costo de penalización / costo de almacenamiento. Cada plan parcial entrega
    solo los productos más críticos: el 100 %, 75 %, 50 % y 25 % del ranking.
    """

    fracciones_ranking: Tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)

    def _planes_de_entrega(self, plan_completo: dict, cliente) -> List[dict]:
        if not plan_completo or sum(plan_completo.values()) == 0:
            return []

        criticidad: Dict[int, float] = {}
        for id_producto in plan_completo:
            costo_penalizacion = cliente.costos_penalizacion.get(id_producto, 0.0)
            costo_almacenamiento = cliente.costos_inventario.get(id_producto, 1.0)
            # Si almacenar es gratis, el producto se considera infinitamente crítico.
            criticidad[id_producto] = float('inf') if costo_almacenamiento <= 0 else costo_penalizacion / costo_almacenamiento

        ranking = sorted(criticidad.keys(), key=lambda id_producto: criticidad[id_producto], reverse=True)

        planes = []
        planes_vistos: Set[Tuple[Tuple[int, int], ...]] = set()
        for fraccion in self.fracciones_ranking:
            productos_incluidos = set(ranking[: math.ceil(fraccion * len(ranking))])
            plan_parcial = {
                id_producto: cantidad
                for id_producto, cantidad in plan_completo.items()
                if id_producto in productos_incluidos and cantidad > 0
            }
            if not plan_parcial:
                continue
            identificador = tuple(sorted(plan_parcial.items()))
            if identificador not in planes_vistos:
                planes_vistos.add(identificador)
                planes.append(plan_parcial)
        return planes
