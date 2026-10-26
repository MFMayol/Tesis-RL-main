"""Rutas del proyecto, independientes del directorio desde el que se ejecute."""
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
CARPETA_INSTANCIAS = RAIZ_PROYECTO / "datos" / "instancias"
CARPETA_RESULTADOS = RAIZ_PROYECTO / "resultados"


def carpeta_politica(clave_politica: str) -> Path:
    return CARPETA_RESULTADOS / clave_politica


def ruta_resultados_politica(clave_politica: str) -> Path:
    return carpeta_politica(clave_politica) / "resultados.csv"


def carpeta_graficos_politica(clave_politica: str) -> Path:
    return carpeta_politica(clave_politica) / "graficos"
