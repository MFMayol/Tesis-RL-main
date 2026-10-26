import os
import time
from typing import List

import pandas as pd

from simulador_irp.codigos_instancia import separar_nombre
from simulador_irp.rutas import ruta_resultados_politica

CODIFICACION_CSV = "utf-8-sig"  # Con marca BOM, Excel muestra bien los acentos al abrir el CSV.

NOMBRES_COLUMNAS = [
    "politica",
    "nombre_politica",
    "familia_politica",
    "tipo_evaluacion",
    "conjunto_instancias",
    "instancia",
    "configuracion_instancia",
    "repeticion",
    "cantidad_clientes",
    "cantidad_vehiculos",
    "cantidad_productos",
    "horizonte_periodos",
    "funcion_objetivo",
    "costo_traslado",
    "costo_demanda_insatisfecha",
    "costo_almacenamiento",
    "porcentaje_traslado",
    "porcentaje_demanda_insatisfecha",
    "porcentaje_almacenamiento",
    "periodos_promedio",
    "desviacion_estandar_funcion_objetivo",
    "simulaciones_evaluacion",
    "episodios_entrenamiento",
    "tiempo_entrenamiento_segundos",
    "tiempo_total_segundos",
    "umbral_inventario_clientes",
    "umbral_inventario_vehiculos",
    "hiperparametros",
    "fecha_ejecucion",
]


class ArchivoResultadosBloqueado(Exception):
    """Un archivo de resultados está abierto en otro programa (normalmente Excel)."""


def _clave_orden_instancia(nombre: str) -> tuple:
    nombre_separado = separar_nombre(nombre)
    return nombre_separado.configuracion or nombre, nombre_separado.repeticion or 0


def leer_resultados(clave_politica: str) -> pd.DataFrame:
    """Resultados guardados de una política; un DataFrame vacío si todavía no tiene."""
    ruta = ruta_resultados_politica(clave_politica)
    if not ruta.exists():
        return pd.DataFrame(columns=NOMBRES_COLUMNAS)
    try:
        return pd.read_csv(ruta, encoding=CODIFICACION_CSV).reindex(columns=NOMBRES_COLUMNAS)
    except PermissionError as error:
        raise ArchivoResultadosBloqueado(str(ruta)) from error


def guardar_resultados(filas: List[dict], reintentos: int = 5) -> None:
    """Guarda cada fila en el CSV de su política; una fila de la misma política e instancia reemplaza a la anterior.

    Se escribe en un archivo temporal y luego se reemplaza el CSV, de modo que un fallo no lo corrompe.

    Raises:
        ArchivoResultadosBloqueado: si algún CSV sigue abierto en otro programa tras los reintentos.
    """
    claves_politicas = list(dict.fromkeys(fila["politica"] for fila in filas))
    for clave in claves_politicas:
        _guardar_politica(clave, [fila for fila in filas if fila["politica"] == clave], reintentos)


def _guardar_politica(clave_politica: str, filas: List[dict], reintentos: int) -> None:
    existentes = leer_resultados(clave_politica)
    # Se combinan registros (no tablas) para que columnas vacías no alteren los tipos de datos.
    combinadas = pd.DataFrame(existentes.to_dict("records") + filas, columns=NOMBRES_COLUMNAS)
    combinadas = combinadas.drop_duplicates(subset="instancia", keep="last")
    combinadas = combinadas.sort_values("instancia", key=lambda nombres: nombres.map(_clave_orden_instancia))
    combinadas = combinadas.reset_index(drop=True)

    ruta = ruta_resultados_politica(clave_politica)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(f"{ruta.stem}.escribiendo.csv")
    combinadas.to_csv(temporal, index=False, encoding=CODIFICACION_CSV)

    for intento in range(reintentos):
        try:
            os.replace(temporal, ruta)
            return
        except PermissionError:
            if intento < reintentos - 1:
                time.sleep(2)
    temporal.unlink(missing_ok=True)
    raise ArchivoResultadosBloqueado(str(ruta))
