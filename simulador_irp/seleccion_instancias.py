import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from simulador_irp.codigos_instancia import separar_nombre
from simulador_irp.rutas import CARPETA_INSTANCIAS, RAIZ_PROYECTO

NOMBRE_CONJUNTO_EXTERNO = "externo"


@dataclass(frozen=True)
class ConjuntoInstancias:
    nombre: str
    ruta: Path
    archivos: Tuple[Path, ...]


def _clave_orden_natural(texto: str) -> list:
    return [int(trozo) if trozo.isdigit() else trozo.casefold() for trozo in re.split(r"(\d+)", texto)]


def ordenar_instancias(rutas: List[Path]) -> List[Path]:
    """Ordena por nombre respetando los números (rep2 antes de rep10)."""
    return sorted(rutas, key=lambda ruta: _clave_orden_natural(ruta.name))


def listar_conjuntos(carpeta: Path = CARPETA_INSTANCIAS) -> List[ConjuntoInstancias]:
    """Subcarpetas de `carpeta` que contienen archivos XML de instancias."""
    conjuntos = []
    for subcarpeta in sorted(carpeta.iterdir(), key=lambda ruta: ruta.name) if carpeta.exists() else []:
        archivos = ordenar_instancias(list(subcarpeta.glob("*.xml"))) if subcarpeta.is_dir() else []
        if archivos:
            conjuntos.append(ConjuntoInstancias(subcarpeta.name, subcarpeta, tuple(archivos)))
    return conjuntos


def agrupar_por_configuracion(archivos) -> Dict[str, List[Path]]:
    """Agrupa archivos por código de configuración (p. ej. 'C2M2P1GU'); los de nombre no estándar van por nombre."""
    grupos: Dict[str, List[Path]] = {}
    for archivo in archivos:
        configuracion = separar_nombre(archivo.stem).configuracion or archivo.stem
        grupos.setdefault(configuracion, []).append(archivo)
    return {configuracion: ordenar_instancias(grupos[configuracion]) for configuracion in sorted(grupos, key=_clave_orden_natural)}


def filtrar_repeticiones(archivos: List[Path], repeticiones: set) -> List[Path]:
    return [archivo for archivo in archivos if separar_nombre(archivo.stem).repeticion in repeticiones]


def resolver_rutas_manuales(texto: str) -> List[Path]:
    """Convierte rutas separadas por ';' (archivos, carpetas o patrones con '*') en archivos XML existentes."""
    encontrados: List[Path] = []
    for fragmento in texto.split(";"):
        patron = fragmento.strip().strip('"')
        if not patron:
            continue
        candidato = Path(patron)
        if not candidato.is_absolute():
            candidato = RAIZ_PROYECTO / candidato
        for coincidencia in map(Path, glob.glob(str(candidato))):
            if coincidencia.is_dir():
                encontrados.extend(ruta.resolve() for ruta in coincidencia.glob("*.xml"))
            elif coincidencia.suffix.casefold() == ".xml":
                encontrados.append(coincidencia.resolve())
    return ordenar_instancias(list(dict.fromkeys(encontrados)))


def nombre_conjunto(ruta: Path) -> str:
    """Nombre de la carpeta de `datos/instancias` que contiene la instancia, o 'externo' si está fuera."""
    try:
        return ruta.resolve().relative_to(CARPETA_INSTANCIAS.resolve()).parts[0]
    except ValueError:
        return NOMBRE_CONJUNTO_EXTERNO
