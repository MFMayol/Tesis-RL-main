import time
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import random

from simulador_irp.nucleo.estado import Estado
from simulador_irp.nucleo.simulador_transicion import ID_DEPOSITO
from simulador_irp.politicas.heuristicas import PoliticaHeuristicaBase
from simulador_irp.politicas.politica_base import ResultadoSimulacion, promediar_resultados
from simulador_irp.utilidades.expansion_fourier import (
    expansion_fourier_interacciones,
    expansion_fourier_unidimensional,
)

UMBRAL_OCUPACION_PARA_ENTREGA = 0.2  # Sobre este nivel de ocupación del cliente no se considera la entrega.
UMBRAL_UTILIZACION_VEHICULO_DEPOSITO = 0.1
DISTANCIA_CERCANA = 500
DISTANCIA_MEDIA = 1000

EPISODIOS_EVALUACION_INICIAL_LINEAL = 10
EPISODIOS_EVALUACION = 20
PERIODO_EVALUACION_EPISODIOS = 100


class PasoEpisodio(NamedTuple):
    caracteristicas: np.ndarray
    recompensa: float


class CaracteristicasCrudas(NamedTuple):
    """Valores normalizados del par (estado, acción), antes de cualquier expansión.

    tiempo_restante: fracción del horizonte que falta.
    ratios_inventario: por (cliente, producto), peso almacenado / capacidad del cliente.
    indicadores_deposito: por (vehículo, producto), 1 si el vehículo vuelve al depósito casi vacío de ese producto.
    entregas: por (cliente, producto), 4 valores: fracción entregada y esa fracción según la distancia del vehículo.
    interacciones: por (cliente, producto), el par (ratio de inventario, fracción entregada).
    """

    tiempo_restante: float
    ratios_inventario: List[float]
    indicadores_deposito: List[float]
    entregas: List[float]
    interacciones: List[Tuple[float, float]]


class AproximadorFuncionValor(PoliticaHeuristicaBase):
    """Política que elige la acción de menor valor estimado Q(s, a) y aprende ese valor por simulación.

    Las subclases definen cómo se estima Q (`seleccionar_mejor_accion`) y cómo se entrena (`entrenar_modelo`).
    Al terminar el entrenamiento, los atributos `mejor_*` describen el mejor modelo encontrado.
    """

    def __init__(self, instancia, simulador, episodios: int, epsilon: float, tasa_aprendizaje: float):
        super().__init__(instancia, simulador)
        self.episodios = episodios
        self.epsilon = epsilon
        self.tasa_aprendizaje = tasa_aprendizaje

        self.mejor_funcion_objetivo = None
        self.mejor_costo_traslado = None
        self.mejor_costo_demanda_insatisfecha = None
        self.mejor_costo_almacenamiento = None
        self.mejores_periodos = None
        self.registro_funcion_objetivo: List[float] = []
        self.registro_mejor_funcion_objetivo: List[float] = []
        self.tiempo_entrenamiento_segundos = None

    def entrenar_modelo(self) -> None:
        raise NotImplementedError

    def seleccionar_mejor_accion(self, estado: Estado, acciones: List[dict]) -> dict:
        raise NotImplementedError

    def _guardar_mejor_modelo(self) -> None:
        raise NotImplementedError

    def tomar_accion(self, estado: Estado) -> dict:
        return self.seleccionar_mejor_accion(estado, self.obtener_acciones(estado))

    # ---- Características ----

    def cantidad_caracteristicas(self) -> int:
        id_clientes = len(self.instancia.id_clientes)
        id_productos = len(self.instancia.id_productos)
        id_vehiculos = len(self.instancia.id_vehiculos)
        # Intercepto, tiempo restante, inventarios, indicadores de depósito y 4 valores de entrega por cliente-producto.
        return 2 + id_clientes * id_productos + id_vehiculos * id_productos + 4 * id_clientes * id_productos

    def caracteristicas(self, estado: Estado, accion: dict) -> np.ndarray:
        """Vector de características lineales del par (estado, acción)."""
        crudas = self._caracteristicas_crudas(estado, accion)
        return np.concatenate([
            np.array([1.0], dtype=np.float32),
            np.array([crudas.tiempo_restante], dtype=np.float32),
            np.array(crudas.ratios_inventario, dtype=np.float32),
            np.array(crudas.indicadores_deposito, dtype=np.float32),
            np.array(crudas.entregas, dtype=np.float32),
        ])

    def _caracteristicas_crudas(self, estado: Estado, accion: dict) -> CaracteristicasCrudas:
        instancia = self.instancia
        clientes = instancia.clientes
        productos = instancia.productos

        planificacion_posterior = {
            id_vehiculo: estado.planificacion.get(id_vehiculo, {}) for id_vehiculo in instancia.id_vehiculos
        }
        for id_vehiculo, plan in accion.items():
            planificacion_posterior[id_vehiculo] = plan

        tiempo_restante = (instancia.horizonte_tiempo - estado.tiempo) / instancia.horizonte_tiempo

        ratios_inventario = []
        ratio_por_par = {}
        for id_cliente in instancia.id_clientes:
            capacidad = clientes[id_cliente].capacidad_almacenamiento
            capacidad = capacidad if capacidad > 0 else 1.0
            for id_producto in instancia.id_productos:
                ratio = (estado.inventarios_clientes[id_cliente][id_producto] * productos[id_producto].peso) / capacidad
                ratios_inventario.append(ratio)
                ratio_por_par[(id_cliente, id_producto)] = ratio

        utilizacion_vehiculos = self._utilizacion_vehiculos(estado)
        indicadores_deposito = []
        for id_vehiculo in instancia.id_vehiculos:
            plan_vehiculo = planificacion_posterior.get(id_vehiculo, {})
            va_al_deposito = bool(plan_vehiculo and next(iter(plan_vehiculo)) == ID_DEPOSITO)
            for id_producto in instancia.id_productos:
                if not plan_vehiculo:
                    indicadores_deposito.append(0.0)
                elif va_al_deposito and utilizacion_vehiculos.get(id_vehiculo, {}).get(id_producto, 0.0) < UMBRAL_UTILIZACION_VEHICULO_DEPOSITO:
                    indicadores_deposito.append(1.0)
                else:
                    indicadores_deposito.append(0.0)

        entregas_por_par = self._entregas_planificadas(estado, planificacion_posterior)

        entregas = []
        interacciones = []
        for id_cliente in instancia.id_clientes:
            capacidad = clientes[id_cliente].capacidad_almacenamiento
            capacidad = capacidad if capacidad > 0 else 1.0
            for id_producto in instancia.id_productos:
                ratio_inventario = ratio_por_par[(id_cliente, id_producto)]
                entrega = entregas_por_par.get((id_cliente, id_producto))
                if ratio_inventario >= UMBRAL_OCUPACION_PARA_ENTREGA or not entrega:
                    entregas.extend([0.0, 0.0, 0.0, 0.0])
                    interacciones.append((ratio_inventario, 0.0))
                    continue

                ratio_entrega = entrega['peso'] / capacidad
                distancia = entrega['distancia']
                entregas.extend([
                    ratio_entrega,
                    ratio_entrega if distancia <= DISTANCIA_CERCANA else 0.0,
                    ratio_entrega if DISTANCIA_CERCANA < distancia <= DISTANCIA_MEDIA else 0.0,
                    ratio_entrega if distancia > DISTANCIA_MEDIA else 0.0,
                ])
                interacciones.append((ratio_inventario, ratio_entrega))

        return CaracteristicasCrudas(tiempo_restante, ratios_inventario, indicadores_deposito, entregas, interacciones)

    def _entregas_planificadas(self, estado: Estado, planificacion_posterior: dict) -> Dict[Tuple[int, int], dict]:
        """Peso planificado y distancia del vehículo para cada par (cliente, producto) con entrega prevista."""
        clientes = self.instancia.clientes
        productos = self.instancia.productos
        entregas = {}
        for id_vehiculo, plan_vehiculo in planificacion_posterior.items():
            if not plan_vehiculo:
                continue
            id_destino = next(iter(plan_vehiculo))
            if id_destino == ID_DEPOSITO or id_vehiculo not in estado.posiciones_vehiculos:
                continue

            posicion_vehiculo = estado.posiciones_vehiculos[id_vehiculo]
            distancia = np.hypot(
                clientes[id_destino].posicion_x - posicion_vehiculo['x'],
                clientes[id_destino].posicion_y - posicion_vehiculo['y'],
            )
            for id_producto, cantidad in plan_vehiculo[id_destino].items():
                if cantidad > 0:
                    peso = cantidad * productos[id_producto].peso
                    if (id_destino, id_producto) not in entregas:
                        entregas[(id_destino, id_producto)] = {'peso': 0.0, 'distancia': distancia}
                    entregas[(id_destino, id_producto)]['peso'] += peso
        return entregas

    def _utilizacion_vehiculos(self, estado: Estado) -> Dict[int, Dict[int, float]]:
        """Peso cargado de cada producto sobre la capacidad del vehículo: {id_vehiculo: {id_producto: razón}}."""
        utilizacion = {}
        for id_vehiculo, inventario in estado.inventarios_vehiculos.items():
            vehiculo = self.instancia.vehiculos.get(id_vehiculo)
            if not vehiculo:
                continue
            capacidad = vehiculo.capacidad if vehiculo.capacidad > 0 else 1.0
            utilizacion[id_vehiculo] = {
                id_producto: cantidad * self.instancia.productos[id_producto].peso / capacidad
                for id_producto, cantidad in inventario.items()
                if id_producto in self.instancia.productos
            }
        return utilizacion

    # ---- Simulación y evaluación ----

    def simular_episodio_greedy(self) -> ResultadoSimulacion:
        """Simula un episodio eligiendo siempre la mejor acción según el modelo actual."""
        estado = self.simulador.determinar_estado_inicial()
        costo_total = costo_traslado = costo_demanda_insatisfecha = costo_almacenamiento = 0
        periodos = 0
        while True:
            accion = self.seleccionar_mejor_accion(estado, self.obtener_acciones(estado))
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

    def ejecutar_episodio_epsilon_greedy(self) -> List[PasoEpisodio]:
        """Episodio de entrenamiento: explora con probabilidad epsilon y registra características y recompensa de cada paso."""
        estado = self.simulador.determinar_estado_inicial()
        pasos = []
        while True:
            accion = self._elegir_accion_epsilon_greedy(estado)
            caracteristicas = self.caracteristicas(estado, accion)
            transicion = self.simulador.transicion(estado, accion)
            pasos.append(PasoEpisodio(caracteristicas, transicion.costo_total))
            if self.es_terminal(transicion.estado_nuevo):
                return pasos
            estado = transicion.estado_nuevo

    def _elegir_accion_epsilon_greedy(self, estado: Estado) -> dict:
        sorteo = np.random.random()
        acciones = self.obtener_acciones(estado)
        if sorteo < self.epsilon:
            return random.choice(acciones)
        return self.seleccionar_mejor_accion(estado, acciones)

    def actualizar_mejor_modelo(self, cantidad_episodios: int = EPISODIOS_EVALUACION) -> None:
        """Evalúa el modelo actual con episodios greedy y lo conserva si mejora la mejor función objetivo."""
        evaluacion = promediar_resultados(self.simular_episodio_greedy() for _ in range(cantidad_episodios))
        self.registro_funcion_objetivo.append(evaluacion.costo_total)

        if self.mejor_funcion_objetivo is None or evaluacion.costo_total < self.mejor_funcion_objetivo:
            self.mejor_funcion_objetivo = evaluacion.costo_total
            self.mejor_costo_traslado = evaluacion.costo_traslado
            self.mejor_costo_demanda_insatisfecha = evaluacion.costo_demanda_insatisfecha
            self.mejor_costo_almacenamiento = evaluacion.costo_almacenamiento
            self.mejores_periodos = evaluacion.periodos
            self._guardar_mejor_modelo()

        self.registro_mejor_funcion_objetivo.append(self.mejor_funcion_objetivo)


class MonteCarloLineal(AproximadorFuncionValor):
    """Monte Carlo on-policy con Q(s, a) lineal en las características, entrenado por descenso de gradiente."""

    def __init__(self, instancia, simulador, episodios, epsilon, tasa_aprendizaje):
        super().__init__(instancia, simulador, episodios, epsilon, tasa_aprendizaje)
        self.coeficientes = None
        self.mejores_coeficientes = None

    def entrenar_modelo(self) -> None:
        self.coeficientes = np.zeros(self.cantidad_caracteristicas(), dtype=np.float32)
        self.mejores_coeficientes = self.coeficientes.copy()
        self.actualizar_mejor_modelo(EPISODIOS_EVALUACION_INICIAL_LINEAL)

        inicio = time.perf_counter()
        for episodio in range(self.episodios):
            retorno = 0
            for paso in reversed(self.ejecutar_episodio_epsilon_greedy()):
                retorno = retorno + paso.recompensa
                self._actualizar_coeficientes(paso.caracteristicas, retorno)

            if episodio % PERIODO_EVALUACION_EPISODIOS == 0:
                self.actualizar_mejor_modelo()
        self.tiempo_entrenamiento_segundos = time.perf_counter() - inicio

    def seleccionar_mejor_accion(self, estado: Estado, acciones: List[dict]) -> dict:
        valores = [(np.dot(self.caracteristicas(estado, accion), self.coeficientes), accion) for accion in acciones]
        return min(valores, key=lambda par: par[0])[1]

    def _guardar_mejor_modelo(self) -> None:
        self.mejores_coeficientes = self.coeficientes.copy()

    def _actualizar_coeficientes(self, caracteristicas: np.ndarray, retorno: float) -> None:
        """Un paso de descenso de gradiente sobre el error cuadrático entre Q(s, a) y el retorno observado."""
        error = np.dot(caracteristicas, self.coeficientes) - retorno
        self.coeficientes -= self.tasa_aprendizaje * (caracteristicas * error)


class MonteCarloFourier(MonteCarloLineal):
    """Monte Carlo con base de Fourier: cada característica y cada par (inventario, entrega) se expanden en senos y cosenos."""

    def __init__(self, instancia, simulador, episodios, epsilon, tasa_aprendizaje, orden_fourier: int):
        super().__init__(instancia, simulador, episodios, epsilon, tasa_aprendizaje)
        self.orden_fourier = orden_fourier
        self.orden_fourier_entrega = 3

    def cantidad_caracteristicas(self) -> int:
        id_clientes = len(self.instancia.id_clientes)
        id_productos = len(self.instancia.id_productos)
        id_vehiculos = len(self.instancia.id_vehiculos)

        caracteristicas_crudas = 1 + id_clientes * id_productos + id_vehiculos * id_productos + 4 * id_clientes * id_productos
        expansion_unidimensional = caracteristicas_crudas * 2 * self.orden_fourier
        expansion_interacciones = id_clientes * id_productos * self.orden_fourier * self.orden_fourier_entrega * 2
        return 1 + expansion_unidimensional + expansion_interacciones  # El 1 inicial es el intercepto.

    def caracteristicas(self, estado: Estado, accion: dict) -> np.ndarray:
        crudas = self._caracteristicas_crudas(estado, accion)
        valores = np.array(
            [crudas.tiempo_restante, *crudas.ratios_inventario, *crudas.indicadores_deposito, *crudas.entregas],
            dtype=np.float32,
        )
        expansion_unidimensional = expansion_fourier_unidimensional(valores, orden=self.orden_fourier)

        inventarios = np.array([par[0] for par in crudas.interacciones], dtype=np.float32)
        entregas = np.array([par[1] for par in crudas.interacciones], dtype=np.float32)
        expansion_interacciones = expansion_fourier_interacciones(
            inventarios, entregas, orden_1=self.orden_fourier, orden_2=self.orden_fourier_entrega
        )
        return np.concatenate(([1.0], expansion_unidimensional, expansion_interacciones))
