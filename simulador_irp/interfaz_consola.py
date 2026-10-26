from pathlib import Path
from typing import List, Sequence, Set, Tuple

import pandas as pd

from simulador_irp.almacen_resultados import ArchivoResultadosBloqueado, guardar_resultados, leer_resultados
from simulador_irp.catalogo_politicas import POLITICAS, EspecificacionPolitica
from simulador_irp.codigos_instancia import describir_configuracion, separar_nombre
from simulador_irp.ejecucion import ejecutar_politica
from simulador_irp.graficos import guardar_grafico_entrenamiento
from simulador_irp.hiperparametros import DEFINICIONES, ValorInvalido, convertir, formatear, resumen, validar
from simulador_irp.rutas import CARPETA_RESULTADOS, RAIZ_PROYECTO, ruta_resultados_politica
from simulador_irp.seleccion_instancias import (
    agrupar_por_configuracion,
    filtrar_repeticiones,
    listar_conjuntos,
    resolver_rutas_manuales,
)

ANCHO = 66


class EntradaCancelada(Exception):
    """El usuario canceló o se cerró la entrada estándar."""


def _leer(texto: str) -> str:
    try:
        return input(texto).strip()
    except EOFError as error:
        raise EntradaCancelada from error


def _es_cancelacion(respuesta: str) -> bool:
    return respuesta.casefold() in {"q", "salir", "cancelar"}


def pedir_politicas() -> List[EspecificacionPolitica]:
    print("\nPOLÍTICAS DISPONIBLES")
    familia_actual = None
    for numero, politica in enumerate(POLITICAS, start=1):
        if politica.familia != familia_actual:
            familia_actual = politica.familia
            print(f"\n  {familia_actual}")
        print(f"   {numero:>2}. {politica.nombre}")
        print(f"       {politica.descripcion}")
    print("\n  Indica uno o varios números separados por coma, 'todas' para ejecutarlas todas o 'q' para salir.")

    while True:
        respuesta = _leer("Políticas: ")
        if _es_cancelacion(respuesta):
            raise EntradaCancelada
        if respuesta.casefold() in {"todas", "todos"}:
            return list(POLITICAS)
        try:
            numeros = [int(trozo) for trozo in respuesta.split(",")]
        except ValueError:
            numeros = []
        if numeros and all(1 <= numero <= len(POLITICAS) for numero in numeros):
            return [POLITICAS[numero - 1] for numero in dict.fromkeys(numeros)]
        print("Opción no válida; ejemplo: 1,4,6")


def pedir_hiperparametros(politicas: Sequence[EspecificacionPolitica]) -> List[EspecificacionPolitica]:
    """Pregunta los hiperparámetros de las políticas de aprendizaje; las demás se devuelven sin cambios."""
    return [_pedir_hiperparametros_politica(politica) if politica.es_aprendizaje else politica for politica in politicas]


def _pedir_hiperparametros_politica(politica: EspecificacionPolitica) -> EspecificacionPolitica:
    print(f"\nHIPERPARÁMETROS: {politica.nombre}")
    print("  Pulsa Enter para conservar el valor por defecto (entre corchetes);")
    print("  escribe 'd' para usar los valores por defecto del resto de esta política.")

    elegidos = dict(politica.parametros)
    usar_por_defecto = False
    for clave, por_defecto in politica.parametros.items():
        if usar_por_defecto:
            try:
                validar(clave, por_defecto, elegidos)
                continue
            except ValorInvalido:
                usar_por_defecto = False

        definicion = DEFINICIONES[clave]
        while True:
            respuesta = _leer(f"  {definicion.etiqueta} [{formatear(por_defecto)}] ({definicion.ayuda}): ")
            if _es_cancelacion(respuesta):
                raise EntradaCancelada
            if respuesta.casefold() in {"d", "defecto"}:
                usar_por_defecto = True
                respuesta = ""
            try:
                valor = por_defecto if respuesta == "" else convertir(clave, respuesta)
                validar(clave, valor, elegidos)
            except ValorInvalido as error:
                usar_por_defecto = False
                print(f"    Valor no válido: {error}.")
                continue
            elegidos[clave] = valor
            break
    return politica.con_parametros(elegidos)


def pedir_instancias() -> List[Path]:
    conjuntos = listar_conjuntos()
    print("\nORIGEN DE LAS INSTANCIAS")
    for numero, conjunto in enumerate(conjuntos, start=1):
        print(f"  {numero}. {conjunto.nombre} ({len(conjunto.archivos)} archivos XML)")
    print("  m. Indicar rutas, carpetas o patrones manualmente")
    print("  q. Cancelar")

    while True:
        respuesta = _leer("Conjunto: ")
        if _es_cancelacion(respuesta):
            raise EntradaCancelada
        if respuesta.casefold() == "m":
            return _pedir_rutas_manuales()
        if respuesta.isdigit() and 1 <= int(respuesta) <= len(conjuntos):
            return _pedir_instancias_del_conjunto(conjuntos[int(respuesta) - 1])
        print("Opción no válida.")


def _pedir_rutas_manuales() -> List[Path]:
    while True:
        respuesta = _leer("Rutas XML separadas por ';' (se admite '*'; 'q' cancela): ")
        if _es_cancelacion(respuesta):
            raise EntradaCancelada
        rutas = resolver_rutas_manuales(respuesta)
        if rutas:
            return rutas
        print("No se encontraron archivos XML; revisa las rutas.")


def _pedir_instancias_del_conjunto(conjunto) -> List[Path]:
    configuraciones = agrupar_por_configuracion(conjunto.archivos)
    nombres = list(configuraciones)

    print(f"\nCONFIGURACIONES DE {conjunto.nombre}")
    print("   0. Todas las instancias del conjunto")
    for numero, configuracion in enumerate(nombres, start=1):
        print(f"  {numero:>2}. {configuracion}  ({len(configuraciones[configuracion])} repeticiones)")
        print(f"      {describir_configuracion(configuracion)}")

    while True:
        respuesta = _leer("Configuración ('q' cancela): ")
        if _es_cancelacion(respuesta):
            raise EntradaCancelada
        if respuesta == "0":
            return list(conjunto.archivos)
        if respuesta.isdigit() and 1 <= int(respuesta) <= len(nombres):
            return _pedir_repeticiones(configuraciones[nombres[int(respuesta) - 1]])
        print("Opción no válida.")


def _pedir_repeticiones(archivos: List[Path]) -> List[Path]:
    disponibles = {separar_nombre(archivo.stem).repeticion for archivo in archivos}
    if len(archivos) <= 1 or None in disponibles:
        return archivos

    print(f"\nRepeticiones disponibles: {', '.join(str(numero) for numero in sorted(disponibles))}")
    while True:
        respuesta = _leer("Repeticiones (números separados por coma, o 'todas'): ")
        if _es_cancelacion(respuesta):
            raise EntradaCancelada
        if respuesta == "" or respuesta.casefold() in {"todas", "todos"}:
            return archivos
        try:
            seleccionadas = {int(trozo) for trozo in respuesta.split(",")}
        except ValueError:
            print("Formato no válido; ejemplo: 1,3,5")
            continue
        if seleccionadas <= disponibles:
            return filtrar_repeticiones(archivos, seleccionadas)
        print(f"No existen las repeticiones: {', '.join(str(n) for n in sorted(seleccionadas - disponibles))}")


def _descripcion_evaluacion(politica: EspecificacionPolitica) -> str:
    if politica.es_aprendizaje:
        return f"entrena {politica.parametros['episodios']:,} episodios por instancia".replace(",", " ")
    return f"promedia {politica.simulaciones} simulaciones por instancia"


def _confirmar_ejecucion(
    politicas: Sequence[EspecificacionPolitica], instancias: Sequence[Path]
) -> List[Tuple[EspecificacionPolitica, Path]]:
    """Muestra el resumen, resuelve resultados ya existentes y devuelve los pares (política, instancia) a ejecutar."""
    print("\n" + "=" * ANCHO)
    print("RESUMEN")
    print("=" * ANCHO)
    for politica in politicas:
        print(f"  {politica.nombre}: {_descripcion_evaluacion(politica)}")
        if politica.es_aprendizaje:
            print(f"    Hiperparámetros: {resumen(politica.parametros)}")
        print(f"    Resultados en {_ruta_relativa(ruta_resultados_politica(politica.clave))}")
    print(f"  Instancias: {len(instancias)}  ({instancias[0].name}{' ...' if len(instancias) > 1 else ''})")

    pares = [(politica, instancia) for politica in politicas for instancia in instancias]
    ya_calculados: Set[Tuple[str, str]] = set()
    try:
        for politica in politicas:
            ya_calculados.update((politica.clave, nombre) for nombre in leer_resultados(politica.clave)["instancia"])
    except ArchivoResultadosBloqueado:
        print("\nUn archivo de resultados está abierto en otro programa. Ciérralo e inténtalo de nuevo.")
        raise EntradaCancelada

    repetidos = [par for par in pares if (par[0].clave, par[1].stem) in ya_calculados]
    if repetidos:
        print(f"\nYa existen resultados para {len(repetidos)} de las {len(pares)} ejecuciones.")
        while True:
            respuesta = _leer("¿[r]eemplazarlos, [o]mitirlos o [c]ancelar? ").casefold()
            if respuesta in {"r", "reemplazar"}:
                break
            if respuesta in {"o", "omitir"}:
                pares = [par for par in pares if par not in repetidos]
                break
            if respuesta in {"c", "cancelar", "q"}:
                raise EntradaCancelada
            print("Responde r, o o c.")

    if not pares:
        print("No queda nada por ejecutar.")
        return []

    print(f"\nSe ejecutarán {len(pares)} combinaciones política-instancia.")
    if _leer("¿Iniciar? [S/n]: ").casefold() in {"n", "no"}:
        raise EntradaCancelada
    return pares


def _ruta_relativa(ruta: Path) -> str:
    return str(ruta.relative_to(RAIZ_PROYECTO))


def _intentar_guardar(filas: List[dict]) -> bool:
    try:
        guardar_resultados(filas)
        return True
    except ArchivoResultadosBloqueado:
        return False


def _guardar_pendientes_al_final(pendientes: List[dict]) -> None:
    """Reintenta guardar las filas que no se pudieron escribir por tener un CSV abierto."""
    while pendientes:
        print("\nNo se pudieron guardar los resultados: algún archivo CSV parece estar abierto en otro programa.")
        try:
            respuesta = _leer("Ciérralo y pulsa Enter para reintentar (o escribe 'copia' para guardar un CSV de respaldo): ")
        except EntradaCancelada:
            respuesta = "copia"
        if respuesta.casefold() == "copia":
            copia = CARPETA_RESULTADOS / "resultados_pendientes.csv"
            pd.DataFrame(pendientes).to_csv(copia, index=False, encoding="utf-8-sig")
            print(f"Resultados guardados en {_ruta_relativa(copia)}")
            return
        if _intentar_guardar(pendientes):
            pendientes.clear()


def _ejecutar_lote(pares: Sequence[Tuple[EspecificacionPolitica, Path]]) -> Tuple[List[dict], int]:
    filas: List[dict] = []
    pendientes: List[dict] = []
    fallos = 0
    csv_bloqueado = False
    total = len(pares)

    try:
        for numero, (politica, instancia) in enumerate(pares, start=1):
            print(f"\n[{numero}/{total}] {politica.nombre} | {instancia.stem}")
            try:
                fila, serie = ejecutar_politica(politica, instancia)
            except Exception as error:
                fallos += 1
                print(f"  ERROR: {type(error).__name__}: {error}")
                continue

            if serie is not None:
                try:
                    guardar_grafico_entrenamiento(politica.clave, instancia.stem, serie)
                except Exception as error:
                    print(f"  Aviso: no se pudo guardar el gráfico ({type(error).__name__}: {error})")

            filas.append(fila)
            pendientes.append(fila)
            if not csv_bloqueado:
                if _intentar_guardar(pendientes):
                    pendientes.clear()
                else:
                    csv_bloqueado = True
                    print("  Aviso: un archivo de resultados está abierto; se guardará al terminar.")
            print(f"  Función objetivo: {fila['funcion_objetivo']:,.2f}  ({fila['tiempo_total_segundos']:.1f} s)")
    except KeyboardInterrupt:
        print("\nEjecución interrumpida por el usuario.")
    finally:
        if pendientes:
            _guardar_pendientes_al_final(pendientes)
    return filas, fallos


def _imprimir_resumen_final(filas: List[dict], fallos: int) -> None:
    print("\n" + "=" * ANCHO)
    print(f"Completadas: {len(filas)} | Con error: {fallos}")
    if filas:
        tabla = pd.DataFrame(filas)[["nombre_politica", "instancia", "funcion_objetivo", "tiempo_total_segundos"]]
        tabla = tabla.rename(columns={
            "nombre_politica": "política", "funcion_objetivo": "función objetivo", "tiempo_total_segundos": "segundos",
        })
        print(tabla.to_string(index=False, float_format=lambda valor: f"{valor:,.1f}"))
        print("\nResultados guardados en:")
        for clave in dict.fromkeys(fila["politica"] for fila in filas):
            print(f"  {_ruta_relativa(ruta_resultados_politica(clave))}")
    print("=" * ANCHO)


def iniciar() -> int:
    print("=" * ANCHO)
    print(" SIMULADOR DE POLÍTICAS DE INVENTARIO Y RUTEO (IRP)")
    print("=" * ANCHO)
    try:
        politicas = pedir_politicas()
        politicas = pedir_hiperparametros(politicas)
        instancias = pedir_instancias()
        pares = _confirmar_ejecucion(politicas, instancias)
    except (EntradaCancelada, KeyboardInterrupt):
        print("\nEjecución cancelada.")
        return 0
    if not pares:
        return 0

    filas, fallos = _ejecutar_lote(pares)
    _imprimir_resumen_final(filas, fallos)
    return 1 if fallos and not filas else 0
