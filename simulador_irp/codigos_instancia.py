import re
from typing import NamedTuple, Optional

_PATRON_NOMBRE = re.compile(
    r"^instancia_(?P<configuracion>C\dM\dP\dG[UC])(?P<sufijo>[A-Za-z]*)_rep(?P<repeticion>\d+)$"
)

_CLIENTES = {"C1": "5-12 clientes", "C2": "13-20 clientes", "C3": "21-28 clientes"}
_VEHICULOS = {
    "M1": "flota pequeña (~1 vehículo cada 4 clientes)",
    "M2": "flota media (~1 vehículo cada 3 clientes)",
    "M3": "flota grande (~1 vehículo cada 2 clientes)",
}
_PRODUCTOS = {"P1": "2 productos", "P2": "4 productos", "P3": "6 productos"}
_GEOGRAFIA = {"GU": "clientes distribuidos uniformemente", "GC": "clientes agrupados en zonas"}


class NombreInstancia(NamedTuple):
    configuracion: Optional[str]
    repeticion: Optional[int]


def separar_nombre(nombre_instancia: str) -> NombreInstancia:
    """Extrae el código de configuración (p. ej. 'C2M2P1GU') y el número de repetición del nombre de una instancia."""
    coincidencia = _PATRON_NOMBRE.match(nombre_instancia)
    if not coincidencia:
        return NombreInstancia(None, None)
    return NombreInstancia(coincidencia["configuracion"], int(coincidencia["repeticion"]))


def describir_configuracion(configuracion: str) -> str:
    """Traduce un código como 'C2M2P1GU' a una descripción legible."""
    coincidencia = re.fullmatch(r"(C\d)(M\d)(P\d)(G[UC])", configuracion)
    if not coincidencia:
        return "configuración personalizada"
    clientes, vehiculos, productos, geografia = coincidencia.groups()
    return ", ".join([_CLIENTES[clientes], _VEHICULOS[vehiculos], _PRODUCTOS[productos], _GEOGRAFIA[geografia]])
