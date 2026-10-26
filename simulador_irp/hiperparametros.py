"""Definición, conversión y validación de los hiperparámetros que el usuario puede elegir."""
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

# Q-learning empieza a aprender al reunir este múltiplo del tamaño del lote en experiencias.
EXPERIENCIAS_MINIMAS_POR_LOTE = 5


class ValorInvalido(ValueError):
    """El texto ingresado no es un valor válido para el hiperparámetro."""


Validador = Callable[[Any, Mapping[str, Any]], None]


@dataclass(frozen=True)
class DefinicionHiperparametro:
    etiqueta: str
    ayuda: str
    convertir: Callable[[str], Any]
    validar: Validador


def _entero(texto: str) -> int:
    try:
        return int(texto.replace(" ", "").replace("_", ""))
    except ValueError:
        raise ValorInvalido("debe ser un número entero, por ejemplo 100") from None


def _decimal(texto: str) -> float:
    try:
        valor = float(texto.replace(" ", "").replace(",", "."))
    except ValueError:
        raise ValorInvalido("debe ser un número, por ejemplo 0.05") from None
    if not math.isfinite(valor):
        raise ValorInvalido("debe ser un número finito")
    return valor


def _lista_enteros(texto: str) -> list:
    partes = [parte for parte in re.split(r"[,;\s]+", texto.strip()) if parte]
    try:
        return [int(parte) for parte in partes]
    except ValueError:
        raise ValorInvalido("usa enteros separados por coma, por ejemplo 64,64") from None


def _entero_minimo(minimo: int) -> Validador:
    def validar(valor, actuales):
        if valor < minimo:
            raise ValorInvalido(f"debe ser al menos {minimo}")
    return validar


def _decimal_entre_cero_y_uno(*, incluye_cero: bool) -> Validador:
    def validar(valor, actuales):
        if not (0 <= valor <= 1) or (valor == 0 and not incluye_cero):
            limite_inferior = "0 (incluido)" if incluye_cero else "0 (excluido)"
            raise ValorInvalido(f"debe estar entre {limite_inferior} y 1 (incluido)")
    return validar


def _capas_validas(valor, actuales) -> None:
    if not valor or any(neuronas < 1 for neuronas in valor):
        raise ValorInvalido("debe haber al menos una capa y cada capa necesita al menos 1 neurona")


def _memoria_suficiente(valor, actuales) -> None:
    minimo = EXPERIENCIAS_MINIMAS_POR_LOTE * actuales["tamano_lote"]
    if valor < minimo:
        raise ValorInvalido(
            f"debe ser al menos {minimo} ({EXPERIENCIAS_MINIMAS_POR_LOTE} veces el tamaño del lote); "
            "con menos, el entrenamiento nunca comienza"
        )


DEFINICIONES: Dict[str, DefinicionHiperparametro] = {
    "episodios": DefinicionHiperparametro(
        "Episodios de entrenamiento", "veces que se simula un episodio completo para entrenar",
        _entero, _entero_minimo(1),
    ),
    "epsilon": DefinicionHiperparametro(
        "Epsilon", "probabilidad de explorar una acción al azar, de 0 a 1",
        _decimal, _decimal_entre_cero_y_uno(incluye_cero=True),
    ),
    "tasa_aprendizaje": DefinicionHiperparametro(
        "Tasa de aprendizaje (learning rate)", "tamaño del paso de actualización, mayor que 0 y hasta 1",
        _decimal, _decimal_entre_cero_y_uno(incluye_cero=False),
    ),
    "orden_fourier": DefinicionHiperparametro(
        "Orden de Fourier", "más orden = más términos de seno y coseno",
        _entero, _entero_minimo(1),
    ),
    "neuronas_por_capa_oculta": DefinicionHiperparametro(
        "Neuronas por capa oculta", "una cifra por capa, por ejemplo 64,64,64",
        _lista_enteros, _capas_validas,
    ),
    "factor_descuento": DefinicionHiperparametro(
        "Factor de descuento", "de 0 a 1; 1 = los costos futuros pesan igual que los inmediatos",
        _decimal, _decimal_entre_cero_y_uno(incluye_cero=True),
    ),
    "tamano_lote": DefinicionHiperparametro(
        "Tamaño del lote", "experiencias por paso de aprendizaje",
        _entero, _entero_minimo(1),
    ),
    "capacidad_memoria_experiencias": DefinicionHiperparametro(
        "Capacidad de la memoria de experiencias", "experiencias recientes que se conservan",
        _entero, _memoria_suficiente,
    ),
}


def convertir(clave: str, texto: str) -> Any:
    """Convierte el texto ingresado al tipo del hiperparámetro."""
    return DEFINICIONES[clave].convertir(texto)


def validar(clave: str, valor: Any, actuales: Mapping[str, Any]) -> None:
    """Valida el valor; `actuales` son los valores ya elegidos de los demás hiperparámetros."""
    DEFINICIONES[clave].validar(valor, actuales)


def formatear(valor: Any) -> str:
    if isinstance(valor, (list, tuple)):
        return ",".join(str(elemento) for elemento in valor)
    if isinstance(valor, float):
        return format(valor, "g")
    return str(valor)


def resumen(parametros: Mapping[str, Any]) -> str:
    return ", ".join(f"{clave}={formatear(valor)}" for clave, valor in parametros.items())
