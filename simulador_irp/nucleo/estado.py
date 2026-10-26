from typing import Dict


class Estado:
    """Estado del sistema en un instante de la simulación.

    Atributos:
        inventarios_clientes: {id_cliente: {id_producto: cantidad}}.
        inventarios_vehiculos: {id_vehiculo: {id_producto: cantidad}}.
        planificacion: {id_vehiculo: {id_destino: {id_producto: cantidad}}}; id_destino 0 es el depósito.
        tiempo: periodo actual de la simulación.
        posiciones_vehiculos: {id_vehiculo: {'x': coordenada_x, 'y': coordenada_y}}.
    """

    def __init__(self):
        self.inventarios_clientes: Dict[int, Dict[int, float]] = {}
        self.inventarios_vehiculos: Dict[int, Dict[int, float]] = {}
        self.planificacion: Dict[int, Dict[int, Dict[int, float]]] = {}
        self.tiempo = 0.0
        self.posiciones_vehiculos: Dict[int, Dict[str, float]] = {}

    def __str__(self):
        return (
            "Estado del sistema:\n"
            f"  Inventarios de clientes: {self.inventarios_clientes}\n"
            f"  Inventarios de vehículos: {self.inventarios_vehiculos}\n"
            f"  Planificación de entregas: {self.planificacion}\n"
            f"  Tiempo transcurrido: {self.tiempo:.2f} periodos\n"
            f"  Posiciones de vehículos: {self.posiciones_vehiculos}"
        )

    def copiar(self) -> 'Estado':
        """Copia profunda del estado; más rápida que copy.deepcopy."""
        copia = Estado()
        copia.inventarios_clientes = {k: v.copy() for k, v in self.inventarios_clientes.items()}
        copia.inventarios_vehiculos = {k: v.copy() for k, v in self.inventarios_vehiculos.items()}
        copia.planificacion = {k: {k2: v2.copy() for k2, v2 in v.items()} for k, v in self.planificacion.items()}
        copia.posiciones_vehiculos = {k: v.copy() for k, v in self.posiciones_vehiculos.items()}
        copia.tiempo = self.tiempo
        return copia
