import json
import time
from datetime import datetime
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

from simulador_irp.catalogo_politicas import EspecificacionPolitica
from simulador_irp.codigos_instancia import separar_nombre
from simulador_irp.nucleo.instancia import Instancia
from simulador_irp.nucleo.simulador_transicion import SimuladorTransicion
from simulador_irp.politicas.politica_base import ResultadoSimulacion
from simulador_irp.seleccion_instancias import nombre_conjunto

PERIODO_REGISTRO_EPISODIOS = 100  # Cada cuántos episodios el entrenamiento registra una evaluación.


def ejecutar_politica(
    especificacion: EspecificacionPolitica,
    ruta_instancia,
    notificar: Callable[[str], None] = print,
) -> Tuple[dict, Optional[pd.DataFrame]]:
    """Ejecuta una política sobre una instancia.

    Returns:
        La fila de resultados (columnas definidas en `almacen_resultados.COLUMNAS`) y, para políticas de
        aprendizaje, la serie de evaluaciones del entrenamiento (None para las de simulación).
    """
    instancia = Instancia(
        ruta_instancia,
        especificacion.umbral_inventario_clientes,
        especificacion.umbral_inventario_vehiculos,
    )
    simulador = SimuladorTransicion(instancia)
    politica = especificacion.crear_politica(instancia, simulador)

    inicio = time.perf_counter()
    if especificacion.es_aprendizaje:
        politica.entrenar_modelo()
        costos = ResultadoSimulacion(
            politica.mejor_funcion_objetivo,
            politica.mejor_costo_traslado,
            politica.mejor_costo_demanda_insatisfecha,
            politica.mejor_costo_almacenamiento,
            politica.mejores_periodos,
        )
        desviacion_funcion_objetivo = None
        serie = _serie_entrenamiento(politica)
    else:
        costos, desviacion_funcion_objetivo = _simular(politica, especificacion.simulaciones, notificar)
        serie = None
    tiempo_total = time.perf_counter() - inicio

    configuracion, repeticion = separar_nombre(instancia.nombre)
    fila = {
        "politica": especificacion.clave,
        "nombre_politica": especificacion.nombre,
        "familia_politica": especificacion.familia,
        "tipo_evaluacion": especificacion.tipo.value,
        "conjunto_instancias": nombre_conjunto(instancia.ruta_archivo),
        "instancia": instancia.nombre,
        "configuracion_instancia": configuracion,
        "repeticion": repeticion,
        "cantidad_clientes": len(instancia.id_clientes),
        "cantidad_vehiculos": len(instancia.id_vehiculos),
        "cantidad_productos": len(instancia.id_productos),
        "horizonte_periodos": instancia.horizonte_tiempo,
        "funcion_objetivo": float(costos.costo_total),
        "costo_traslado": float(costos.costo_traslado),
        "costo_demanda_insatisfecha": float(costos.costo_demanda_insatisfecha),
        "costo_almacenamiento": float(costos.costo_almacenamiento),
        "porcentaje_traslado": _porcentaje(costos.costo_traslado, costos.costo_total),
        "porcentaje_demanda_insatisfecha": _porcentaje(costos.costo_demanda_insatisfecha, costos.costo_total),
        "porcentaje_almacenamiento": _porcentaje(costos.costo_almacenamiento, costos.costo_total),
        "periodos_promedio": float(costos.periodos),
        "desviacion_estandar_funcion_objetivo": desviacion_funcion_objetivo,
        "simulaciones_evaluacion": None if especificacion.es_aprendizaje else especificacion.simulaciones,
        "episodios_entrenamiento": especificacion.parametros.get("episodios") if especificacion.es_aprendizaje else None,
        "tiempo_entrenamiento_segundos": politica.tiempo_entrenamiento_segundos if especificacion.es_aprendizaje else None,
        "tiempo_total_segundos": tiempo_total,
        "umbral_inventario_clientes": especificacion.umbral_inventario_clientes,
        "umbral_inventario_vehiculos": especificacion.umbral_inventario_vehiculos,
        "hiperparametros": json.dumps(dict(especificacion.parametros), ensure_ascii=False),
        "fecha_ejecucion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return fila, serie


def _simular(politica, cantidad: int, notificar) -> Tuple[ResultadoSimulacion, float]:
    resultados = []
    paso_aviso = max(1, cantidad // 10)
    for numero in range(1, cantidad + 1):
        resultados.append(politica.simular_episodio())
        if numero % paso_aviso == 0 or numero == cantidad:
            notificar(f"    simulación {numero}/{cantidad}")

    promedio = ResultadoSimulacion(*(np.mean([resultado[campo] for resultado in resultados]) for campo in range(5)))
    desviacion = float(np.std([resultado.costo_total for resultado in resultados]))
    return promedio, desviacion


def _serie_entrenamiento(politica) -> pd.DataFrame:
    """Evaluaciones registradas durante el entrenamiento, con los episodios entrenados al momento de cada una."""
    evaluadas = list(politica.registro_funcion_objetivo)
    mejores = list(politica.registro_mejor_funcion_objetivo)
    # La primera evaluación es previa al entrenamiento; las siguientes ocurren tras el episodio 0, 100, 200...
    episodios = [0] + [1 + PERIODO_REGISTRO_EPISODIOS * indice for indice in range(len(evaluadas) - 1)]
    return pd.DataFrame({
        "episodios_entrenados": episodios,
        "funcion_objetivo_evaluada": evaluadas,
        "mejor_funcion_objetivo": mejores,
    })


def _porcentaje(parte: float, total: float) -> float:
    return float(parte) / float(total) * 100 if total else 0.0
