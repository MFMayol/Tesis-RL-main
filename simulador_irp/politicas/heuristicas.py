from typing import List, Optional

import numpy as np

from simulador_irp.nucleo.estado import Estado
from simulador_irp.nucleo.simulador_transicion import ID_DEPOSITO
from simulador_irp.politicas.politica_base import PoliticaBase
from simulador_irp.utilidades.agrupamiento import agrupar_clientes_por_vehiculo

CANTIDAD_CLIENTES_CRITICOS = 4
CANTIDAD_VEHICULOS_CERCANOS = 2


class PoliticaHeuristicaBase(PoliticaBase):
    """Reglas de apoyo comunes: qué vehículos y clientes están disponibles, cuáles son críticos y qué acciones existen."""

    def vehiculos_con_inventario_bajo(self, estado: Estado) -> List[int]:
        """Vehículos cuya carga está bajo el umbral de inventario configurado en la instancia."""
        productos = self.instancia.productos
        umbral = self.instancia.umbral_inventario_vehiculos
        return [
            id_vehiculo
            for id_vehiculo, vehiculo in self.instancia.vehiculos.items()
            if sum(cantidad * productos[id_producto].peso for id_producto, cantidad in estado.inventarios_vehiculos[id_vehiculo].items())
            < umbral * vehiculo.capacidad
        ]

    def vehiculos_disponibles(self, estado: Estado) -> List[int]:
        """Vehículos sin planificación."""
        return [id_vehiculo for id_vehiculo in self.instancia.id_vehiculos if not estado.planificacion[id_vehiculo]]

    def clientes_sin_asignar(self, estado: Estado) -> List[int]:
        """Clientes que ningún vehículo tiene planificado visitar."""
        clientes_asignados = set()
        for planificacion in estado.planificacion.values():
            clientes_asignados.update(planificacion.keys())
        return list(set(self.instancia.clientes.keys()) - clientes_asignados)

    def cliente_mas_critico(self, estado: Estado, clientes_candidatos: List[int]) -> Optional[int]:
        """Cliente con la menor cobertura, medida como inventario / demanda media del producto más escaso."""
        id_productos = list(self.instancia.productos.keys())
        demandas = np.array([self.instancia.demandas_medias[id_producto] for id_producto in id_productos], dtype=np.float32)

        cliente_critico = None
        menor_cobertura = np.inf
        for id_cliente in clientes_candidatos:
            inventario = np.array(
                [estado.inventarios_clientes[id_cliente][id_producto] for id_producto in id_productos], dtype=np.float32
            )
            if not np.any(inventario >= 0):
                continue

            coberturas = np.divide(inventario, demandas, out=np.full_like(inventario, np.nan), where=demandas != 0)
            if np.isnan(coberturas).all():
                continue
            cobertura_minima = np.nanmin(coberturas)

            if cobertura_minima < menor_cobertura:
                menor_cobertura = cobertura_minima
                cliente_critico = id_cliente
        return cliente_critico

    def clientes_mas_criticos(
        self, estado: Estado, clientes_candidatos: List[int], cantidad: int = CANTIDAD_CLIENTES_CRITICOS
    ) -> List[int]:
        """Los clientes con menor fracción de su capacidad de almacenamiento ocupada."""
        ocupacion_por_cliente = []
        for id_cliente, inventario in estado.inventarios_clientes.items():
            if id_cliente in clientes_candidatos:
                peso_almacenado = sum(
                    cantidad_producto * self.instancia.productos[id_producto].peso
                    for id_producto, cantidad_producto in inventario.items()
                )
                ocupacion_por_cliente.append(
                    (id_cliente, peso_almacenado / self.instancia.clientes[id_cliente].capacidad_almacenamiento)
                )

        ocupacion_por_cliente.sort(key=lambda par: par[1])
        return [id_cliente for id_cliente, _ in ocupacion_por_cliente[:cantidad]]

    def vehiculos_mas_cercanos(
        self,
        estado: Estado,
        id_cliente: int,
        vehiculos_disponibles: List[int],
        cantidad: int = CANTIDAD_VEHICULOS_CERCANOS,
    ) -> List[int]:
        """Los vehículos disponibles más cercanos al cliente."""
        if not vehiculos_disponibles:
            return []

        cliente = self.instancia.clientes[id_cliente]
        posicion_cliente = np.array([cliente.posicion_x, cliente.posicion_y])
        posiciones_vehiculos = np.array(
            [
                [estado.posiciones_vehiculos[id_vehiculo]['x'], estado.posiciones_vehiculos[id_vehiculo]['y']]
                for id_vehiculo in vehiculos_disponibles
            ],
            dtype=np.float32,
        )
        distancias_cuadradas = np.sum((posiciones_vehiculos - posicion_cliente) ** 2, axis=1)
        indices_mas_cercanos = np.argsort(distancias_cuadradas)[: min(cantidad, len(vehiculos_disponibles))]
        return np.array(vehiculos_disponibles)[indices_mas_cercanos].tolist()

    def obtener_acciones(self, estado: Estado) -> List[dict]:
        """Acciones candidatas: no hacer nada, enviar un vehículo con poca carga al depósito o atender a un cliente crítico."""
        acciones = [{}]

        vehiculos_disponibles = self.vehiculos_disponibles(estado)
        con_inventario_bajo = set(self.vehiculos_con_inventario_bajo(estado))
        vehiculo_a_deposito = next((id_vehiculo for id_vehiculo in vehiculos_disponibles if id_vehiculo in con_inventario_bajo), None)
        if vehiculo_a_deposito is not None:
            acciones.append({vehiculo_a_deposito: {ID_DEPOSITO: {}}})

        clientes_criticos = self.clientes_mas_criticos(estado, self.clientes_sin_asignar(estado))
        for id_cliente in clientes_criticos:
            cliente = self.instancia.clientes[id_cliente]
            for id_vehiculo in self.vehiculos_mas_cercanos(estado, id_cliente, vehiculos_disponibles):
                plan_completo = self.planificar_entrega(estado.inventarios_vehiculos[id_vehiculo], cliente.demanda_media)
                for plan in self._planes_de_entrega(plan_completo, cliente):
                    acciones.append({id_vehiculo: {id_cliente: plan}})
        return acciones

    def _planes_de_entrega(self, plan_completo: dict, cliente) -> List[dict]:
        """Variantes de entrega a evaluar para un vehículo y un cliente; por defecto, solo la entrega completa."""
        return [plan_completo] if sum(plan_completo.values()) > 0 else []


class PoliticaHeuristicaSimple(PoliticaHeuristicaBase):
    """Un vehículo con poca carga vuelve al depósito; si no, el primero disponible visita al cliente más crítico."""

    def tomar_accion(self, estado: Estado) -> dict:
        vehiculos_disponibles = self.vehiculos_disponibles(estado)
        if not vehiculos_disponibles:
            return {}

        con_inventario_bajo = self.vehiculos_con_inventario_bajo(estado)
        for id_vehiculo in vehiculos_disponibles:
            if id_vehiculo in con_inventario_bajo:
                return {id_vehiculo: {ID_DEPOSITO: {}}}

        clientes_sin_asignar = self.clientes_sin_asignar(estado)
        for id_vehiculo in vehiculos_disponibles:
            clientes_candidatos = self._clientes_candidatos(id_vehiculo, clientes_sin_asignar)
            if not clientes_candidatos:
                continue
            id_cliente = self.cliente_mas_critico(estado, clientes_candidatos)
            if id_cliente is None:
                continue
            plan = self._planificar_entrega_vehiculo(estado, id_vehiculo, id_cliente)
            if sum(plan.values()) > 0:
                return {id_vehiculo: {id_cliente: plan}}
        return {}

    def _clientes_candidatos(self, id_vehiculo: int, clientes_sin_asignar: List[int]) -> List[int]:
        """Clientes que el vehículo puede atender; en la política simple, cualquiera sin asignar."""
        return clientes_sin_asignar

    def _planificar_entrega_vehiculo(self, estado: Estado, id_vehiculo: int, id_cliente: int) -> dict:
        return self.planificar_entrega(
            estado.inventarios_vehiculos[id_vehiculo], self.instancia.clientes[id_cliente].demanda_media
        )


class PoliticaHeuristicaClusterizada(PoliticaHeuristicaSimple):
    """Como la política simple, pero cada vehículo solo atiende a los clientes de su grupo k-means."""

    def __init__(self, instancia, simulador):
        super().__init__(instancia, simulador)
        posiciones_clientes = {
            id_cliente: {'x': cliente.posicion_x, 'y': cliente.posicion_y}
            for id_cliente, cliente in instancia.clientes.items()
        }
        self.clientes_por_vehiculo = agrupar_clientes_por_vehiculo(len(instancia.vehiculos), posiciones_clientes)

    def _clientes_candidatos(self, id_vehiculo: int, clientes_sin_asignar: List[int]) -> List[int]:
        clientes_del_vehiculo = self.clientes_por_vehiculo.get(id_vehiculo, [])
        return [id_cliente for id_cliente in clientes_sin_asignar if id_cliente in clientes_del_vehiculo]
