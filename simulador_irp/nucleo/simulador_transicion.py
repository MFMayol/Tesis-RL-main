import math
import random
from typing import NamedTuple, Tuple

from simulador_irp.nucleo.estado import Estado
from simulador_irp.nucleo.instancia import Instancia

ID_DEPOSITO = 0
PERIODOS_MAXIMOS_SIN_NUEVO_ESTADO = 5


class ResultadoTransicion(NamedTuple):
    estado_nuevo: Estado
    costo_total: float
    costo_traslado: float
    costo_demanda_insatisfecha: float
    costo_almacenamiento: float


class SimuladorTransicion:
    """Aplica una acción a un estado y avanza la simulación hasta el siguiente estado de decisión.

    Un nuevo estado de decisión se genera cuando un vehículo llega a su destino, el inventario de un cliente
    cae bajo el umbral, se alcanza el horizonte de tiempo o pasan 5 periodos sin eventos.
    """

    def __init__(self, instancia: Instancia):
        self.instancia = instancia

    def determinar_estado_inicial(self) -> Estado:
        """Estado en el periodo 0: vehículos en el depósito, sin planificación y con los inventarios de la instancia."""
        estado = Estado()
        for id_vehiculo, vehiculo in self.instancia.vehiculos.items():
            estado.inventarios_vehiculos[id_vehiculo] = vehiculo.inventario.copy()
            estado.posiciones_vehiculos[id_vehiculo] = {'x': self.instancia.deposito_x, 'y': self.instancia.deposito_y}
            estado.planificacion[id_vehiculo] = {}

        for id_cliente, cliente in self.instancia.clientes.items():
            estado.inventarios_clientes[id_cliente] = cliente.inventarios_iniciales.copy()

        estado.tiempo = 0
        return estado

    def transicion(self, estado_original: Estado, accion) -> ResultadoTransicion:
        """Aplica la acción y simula el paso del tiempo hasta el siguiente estado de decisión.

        Args:
            estado_original: estado desde el cual se decide.
            accion: {id_vehiculo: {id_destino: {id_producto: cantidad}}}; reemplaza la planificación de esos vehículos.

        Returns:
            Estado nuevo y costos incurridos (total, traslado, demanda insatisfecha y almacenamiento).
        """
        estado = estado_original.copiar()
        instancia = self.instancia

        costo_demanda_insatisfecha = 0
        costo_almacenamiento = 0
        costo_traslado = 0
        hay_nuevo_estado = False
        periodos_transcurridos = 0

        for id_vehiculo, planificacion in accion.items():
            estado.planificacion[id_vehiculo] = planificacion

        pesos_productos = {id_producto: instancia.productos[id_producto].peso for id_producto in instancia.id_productos}
        posicion_deposito = (instancia.deposito_x, instancia.deposito_y)

        while True:
            estado.tiempo += 1
            periodos_transcurridos += 1

            for id_vehiculo, vehiculo in instancia.vehiculos.items():
                if not estado.planificacion[id_vehiculo]:
                    continue

                posicion_actual = (
                    estado.posiciones_vehiculos[id_vehiculo]['x'],
                    estado.posiciones_vehiculos[id_vehiculo]['y'],
                )
                id_destino = next(iter(estado.planificacion[id_vehiculo]))
                velocidad = abs(random.gauss(vehiculo.velocidad_media, vehiculo.desviacion_velocidad))

                if id_destino == ID_DEPOSITO:
                    posicion_destino = posicion_deposito
                    nueva_posicion = self._avanzar_hacia(posicion_actual, posicion_destino, velocidad)
                    estado.posiciones_vehiculos[id_vehiculo]['x'], estado.posiciones_vehiculos[id_vehiculo]['y'] = nueva_posicion
                    if nueva_posicion == posicion_destino:
                        hay_nuevo_estado = True
                        estado.inventarios_vehiculos[id_vehiculo] = vehiculo.inventario.copy()
                        estado.planificacion[id_vehiculo] = {}
                else:
                    cliente_destino = instancia.clientes[id_destino]
                    posicion_destino = (cliente_destino.posicion_x, cliente_destino.posicion_y)
                    nueva_posicion = self._avanzar_hacia(posicion_actual, posicion_destino, velocidad)
                    estado.posiciones_vehiculos[id_vehiculo]['x'], estado.posiciones_vehiculos[id_vehiculo]['y'] = nueva_posicion

                    if nueva_posicion == posicion_destino:
                        hay_nuevo_estado = True
                        self._entregar_al_cliente(estado, id_vehiculo, cliente_destino, pesos_productos)
                        estado.planificacion[id_vehiculo] = {}

            for id_cliente in instancia.id_clientes:
                cliente = instancia.clientes[id_cliente]
                inventarios_cliente = estado.inventarios_clientes[id_cliente]
                for id_producto in instancia.id_productos:
                    demanda = int(abs(random.gauss(cliente.demanda_media[id_producto], cliente.demanda_desviacion[id_producto])))
                    inventario = inventarios_cliente[id_producto]

                    if demanda > inventario:
                        costo_demanda_insatisfecha += (demanda - inventario) * cliente.costos_penalizacion[id_producto]
                        inventarios_cliente[id_producto] = 0
                    else:
                        costo_almacenamiento += (inventario - demanda) * cliente.costos_inventario[id_producto]
                        inventarios_cliente[id_producto] = inventario - demanda

                peso_almacenado = sum(
                    estado.inventarios_clientes[id_cliente][id_producto] * pesos_productos[id_producto]
                    for id_producto in instancia.id_productos
                )
                if peso_almacenado < cliente.capacidad_almacenamiento * instancia.umbral_inventario_clientes:
                    hay_nuevo_estado = True

            if estado.tiempo >= instancia.horizonte_tiempo:
                for id_vehiculo in instancia.id_vehiculos:
                    estado.planificacion[id_vehiculo] = {ID_DEPOSITO: {}}
                hay_nuevo_estado = True

            if periodos_transcurridos >= PERIODOS_MAXIMOS_SIN_NUEVO_ESTADO:
                hay_nuevo_estado = True

            if hay_nuevo_estado:
                break

        for id_vehiculo in instancia.id_vehiculos:
            posicion_anterior = estado_original.posiciones_vehiculos[id_vehiculo]
            posicion_nueva = estado.posiciones_vehiculos[id_vehiculo]
            costo_traslado += math.hypot(
                posicion_anterior['x'] - posicion_nueva['x'],
                posicion_anterior['y'] - posicion_nueva['y'],
            )

        costo_total = costo_traslado + costo_demanda_insatisfecha + costo_almacenamiento
        return ResultadoTransicion(estado, costo_total, costo_traslado, costo_demanda_insatisfecha, costo_almacenamiento)

    def _entregar_al_cliente(self, estado: Estado, id_vehiculo: int, cliente, pesos_productos) -> None:
        """Descarga la entrega planificada; si no cabe en el cliente, se entrega solo la fracción que cabe."""
        entrega = estado.planificacion[id_vehiculo][cliente.id]
        peso_entrega = sum(cantidad * pesos_productos[id_producto] for id_producto, cantidad in entrega.items())
        peso_almacenado = sum(
            estado.inventarios_clientes[cliente.id][id_producto] * pesos_productos[id_producto]
            for id_producto in self.instancia.id_productos
        )

        if peso_entrega <= cliente.capacidad_almacenamiento - peso_almacenado:
            for id_producto, cantidad in entrega.items():
                estado.inventarios_vehiculos[id_vehiculo][id_producto] -= cantidad
                estado.inventarios_clientes[cliente.id][id_producto] += cantidad
            return

        if peso_entrega > 0:
            espacio_disponible = max(0, cliente.capacidad_almacenamiento - peso_almacenado)
            fraccion_entregable = espacio_disponible / peso_entrega
        else:
            fraccion_entregable = 0
        for id_producto, cantidad in entrega.items():
            cantidad_entregada = int(cantidad * fraccion_entregable)
            estado.inventarios_vehiculos[id_vehiculo][id_producto] -= cantidad_entregada
            estado.inventarios_clientes[cliente.id][id_producto] += cantidad_entregada

    @staticmethod
    def _avanzar_hacia(posicion_inicial: Tuple[float, float], destino: Tuple[float, float], distancia_maxima: float):
        """Mueve un vehículo en línea recta hacia el destino sin sobrepasarlo."""
        x_inicial, y_inicial = posicion_inicial
        x_destino, y_destino = destino
        delta_x = x_destino - x_inicial
        delta_y = y_destino - y_inicial
        distancia_total = math.hypot(delta_x, delta_y)

        if distancia_total == 0:
            return x_inicial, y_inicial
        if distancia_maxima >= distancia_total:
            return x_destino, y_destino

        fraccion = distancia_maxima / distancia_total
        return x_inicial + delta_x * fraccion, y_inicial + delta_y * fraccion
