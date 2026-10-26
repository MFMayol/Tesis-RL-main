import numpy as np
from scipy.stats import norm

from simulador_irp.nucleo.estado import Estado
from simulador_irp.politicas.heuristicas import PoliticaHeuristicaSimple


class PoliticaEOQEstocastico(PoliticaHeuristicaSimple):
    """Política (Q, R) estocástica dinámica.

    La cantidad a entregar Q y el punto de reorden R se calculan en tiempo real: el tiempo de entrega y el costo
    de preparación dependen de la distancia actual entre el vehículo y el cliente.
    """

    def __init__(self, instancia, simulador, iteraciones_maximas: int = 15, tolerancia: float = 1e-3):
        super().__init__(instancia, simulador)
        self.iteraciones_maximas = iteraciones_maximas
        self.tolerancia = tolerancia

    @staticmethod
    def funcion_perdida_normal(z: float) -> float:
        """Función de pérdida normal estándar L(z)."""
        return norm.pdf(z) - z * (1 - norm.cdf(z))

    def _planificar_entrega_vehiculo(self, estado: Estado, id_vehiculo: int, id_cliente: int) -> dict:
        """Calcula Q* y R* por producto iterando hasta converger, con tiempo de entrega y costo de preparación reales."""
        cliente = self.instancia.clientes[id_cliente]
        vehiculo = self.instancia.vehiculos[id_vehiculo]
        inventario_vehiculo = estado.inventarios_vehiculos[id_vehiculo]
        inventario_cliente = estado.inventarios_clientes[id_cliente]

        posicion_vehiculo = np.array(
            [estado.posiciones_vehiculos[id_vehiculo]['x'], estado.posiciones_vehiculos[id_vehiculo]['y']]
        )
        posicion_cliente = np.array([cliente.posicion_x, cliente.posicion_y])
        distancia = np.linalg.norm(posicion_vehiculo - posicion_cliente)
        velocidad = max(vehiculo.velocidad_media, 1.0)

        tiempo_entrega = max(distancia / velocidad, 0.05)
        costo_preparacion = max(distancia, 1.0)  # La distancia es un proxy del costo de iniciar un pedido.

        plan_entrega = {}
        for id_producto in self.instancia.productos:
            demanda_media = cliente.demanda_media[id_producto]
            desviacion_demanda = cliente.demanda_desviacion[id_producto]
            costo_mantenimiento = max(cliente.costos_inventario[id_producto], 0.001)
            costo_penalizacion = cliente.costos_penalizacion[id_producto]

            if demanda_media <= 0:
                plan_entrega[id_producto] = 0
                continue

            demanda_en_tiempo_entrega = demanda_media * tiempo_entrega
            desviacion_en_tiempo_entrega = desviacion_demanda * np.sqrt(tiempo_entrega)

            # Se parte del EOQ clásico y se itera con la condición de optimalidad P(demanda > R) = h*Q / (p*D).
            cantidad_pedido = np.sqrt((2 * costo_preparacion * demanda_media) / costo_mantenimiento)
            punto_reorden = 0
            cantidad_pedido_anterior = -1.0

            for _ in range(self.iteraciones_maximas):
                if abs(cantidad_pedido - cantidad_pedido_anterior) < self.tolerancia:
                    break
                cantidad_pedido_anterior = cantidad_pedido

                if costo_penalizacion * demanda_media > 0:
                    probabilidad_quiebre = (costo_mantenimiento * cantidad_pedido) / (costo_penalizacion * demanda_media)
                else:
                    probabilidad_quiebre = 0.0
                probabilidad_quiebre = min(max(probabilidad_quiebre, 1e-6), 1.0 - 1e-6)

                z = norm.ppf(1 - probabilidad_quiebre)
                punto_reorden = demanda_en_tiempo_entrega + z * desviacion_en_tiempo_entrega

                deficit_esperado = desviacion_en_tiempo_entrega * self.funcion_perdida_normal(z)
                costo_ajustado = costo_preparacion + (costo_penalizacion * deficit_esperado)
                cantidad_pedido = np.sqrt((2 * demanda_media * costo_ajustado) / costo_mantenimiento)

            # Un vehículo lejano implica un R grande y se pide antes; uno cercano permite esperar más.
            if inventario_cliente[id_producto] < punto_reorden:
                cantidad_real = min(cantidad_pedido, inventario_vehiculo.get(id_producto, 0))
                plan_entrega[id_producto] = max(0, int(cantidad_real))
            else:
                plan_entrega[id_producto] = 0

        return plan_entrega
