"""Genera las instancias IRP (XML) de los conjuntos de datos del proyecto.

Uso, desde la raíz del proyecto:
    python herramientas/generar_instancias.py                    # regenera ambos conjuntos en datos/instancias
    python herramientas/generar_instancias.py --salida OTRA_RUTA  # escribe en otra carpeta (p. ej. para comparar)

Código de configuración (p. ej. C2M2P1GU): C = clientes, M = vehículos, P = productos, G = geografía
(GU: uniforme, GC: agrupada).
"""
import argparse
import math
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Sequence, Tuple
from xml.dom import minidom

import numpy as np

CARPETA_INSTANCIAS = Path(__file__).resolve().parent.parent / "datos" / "instancias"

RANGO_CLIENTES: Dict[str, Tuple[int, int]] = {"C1": (5, 12), "C2": (13, 20), "C3": (21, 28)}
CANTIDAD_PRODUCTOS: Dict[str, int] = {"P1": 2, "P2": 4, "P3": 6}

ANCHO_ZONA = 1000
LARGO_ZONA = 1000
HORIZONTE_TIEMPO = 600

CONFIGURACIONES_SIN_COSTO_ALMACENAMIENTO = (
    "C1M2P1GU", "C2M2P1GU", "C3M2P1GU", "C2M1P1GU", "C2M3P1GU", "C2M2P2GC", "C2M2P2GU", "C2M2P3GU",
)
CONFIGURACIONES_CON_COSTO_ALMACENAMIENTO = ("C2M2P2GU",)


def cantidad_vehiculos(codigo_vehiculos: str, cantidad_clientes: int) -> int:
    if codigo_vehiculos == "M1":
        cantidad = math.floor(cantidad_clientes / 4)
    elif codigo_vehiculos == "M2":
        cantidad = max(math.floor(cantidad_clientes / 3), math.floor(cantidad_clientes / 4) + 1)
    else:  # M3
        cantidad = max(math.floor(cantidad_clientes / 2), math.floor(cantidad_clientes / 3) + 1)
    return cantidad or 1


def generar_instancia(
    ancho_zona: int,
    largo_zona: int,
    horizonte_tiempo: int,
    cantidad_productos: int,
    cantidad_clientes: int,
    cantidad_vehiculos_flota: int,
    semilla: int,
    carpeta_salida: Path,
    nombre_archivo: str,
    distribucion_geografica: str,
    con_costo_almacenamiento: bool,
) -> Path:
    """Escribe una instancia XML reproducible a partir de la semilla.

    El orden de las llamadas al generador aleatorio define el contenido: no reordenarlas.
    """
    random.seed(semilla)
    np.random.seed(semilla)

    raiz = ET.Element("InstanciaIRP")

    informacion_general = ET.SubElement(raiz, "InformacionGeneral")
    ET.SubElement(informacion_general, "AnchoZona").text = str(ancho_zona)
    ET.SubElement(informacion_general, "LargoZona").text = str(largo_zona)
    ET.SubElement(informacion_general, "Depot_x").text = str(ancho_zona / 2)
    ET.SubElement(informacion_general, "Depot_y").text = str(largo_zona / 2)
    ET.SubElement(informacion_general, "HorizonteTiempo").text = str(horizonte_tiempo)

    productos = ET.SubElement(raiz, "Productos")
    pesos_productos = []
    for indice in range(cantidad_productos):
        producto = ET.SubElement(productos, "Producto")
        producto.set("id", str(indice + 1))
        peso = random.choice([0.5, 1, 1.5, 2])
        ET.SubElement(producto, "Peso").text = str(peso)
        pesos_productos.append(peso)

    clientes = ET.SubElement(raiz, "Clientes")

    centroides = []
    cantidad_centroides = 0
    if distribucion_geografica == "GC":
        cantidad_centroides = max(math.floor(cantidad_clientes / 5), 1)
        for _ in range(cantidad_centroides):
            centroides.append((random.randint(0, ancho_zona), random.randint(0, largo_zona)))

    for indice in range(cantidad_clientes):
        cliente = ET.SubElement(clientes, "Cliente")
        cliente.set("id", str(indice + 1))

        posicion_x, posicion_y = 0, 0
        if distribucion_geografica == "GU":
            posicion_x = random.randint(0, ancho_zona)
            posicion_y = random.randint(0, largo_zona)
        elif distribucion_geografica == "GC":
            centroide = centroides[indice % cantidad_centroides]
            desplazamiento_x = random.randint(-100, 100)
            desplazamiento_y = random.randint(-100, 100)
            posicion_x = max(0, min(ancho_zona, centroide[0] + desplazamiento_x))
            posicion_y = max(0, min(largo_zona, centroide[1] + desplazamiento_y))

        ET.SubElement(cliente, "PosicionX").text = str(posicion_x)
        ET.SubElement(cliente, "PosicionY").text = str(posicion_y)

        capacidad = int(random.uniform(50, 150) * cantidad_productos)
        ET.SubElement(cliente, "CapacidadAlmacenamiento").text = str(capacidad)

        inventarios = ET.SubElement(cliente, "Inventarios")
        costos = ET.SubElement(cliente, "Costos")
        demandas = ET.SubElement(cliente, "Demandas")

        # Cada cliente parte con entre 10 % y 40 % de su bodega ocupada, repartida al azar entre los productos.
        capacidad_ocupada = capacidad * random.uniform(0.1, 0.4)
        proporciones = [random.uniform(0.5, 1.5) for _ in range(cantidad_productos)]
        suma_proporciones = sum(proporciones)

        for indice_producto in range(cantidad_productos):
            capacidad_producto = capacidad_ocupada * (proporciones[indice_producto] / suma_proporciones)
            inventario_inicial = int(capacidad_producto / pesos_productos[indice_producto])

            ET.SubElement(inventarios, f"Producto{indice_producto + 1}").text = str(inventario_inicial)
            costo_penalizacion = np.random.uniform(1, 120)
            costo_inventario = np.random.uniform(1, 3) if con_costo_almacenamiento else 0
            costos_producto = ET.SubElement(costos, f"Producto{indice_producto + 1}")
            ET.SubElement(costos_producto, "CostoInventario").text = str(round(costo_inventario))
            ET.SubElement(costos_producto, "CostoPenalizacion").text = str(round(costo_penalizacion))

            demanda_producto = ET.SubElement(demandas, f"Producto{indice_producto + 1}")
            ET.SubElement(demanda_producto, "Media").text = str(round(random.uniform(0, 3), 2))
            ET.SubElement(demanda_producto, "DesvEst").text = str(round(random.uniform(0, 1), 2))

    vehiculos = ET.SubElement(raiz, "Vehiculos")
    for indice in range(cantidad_vehiculos_flota):
        vehiculo = ET.SubElement(vehiculos, "Vehiculo")
        vehiculo.set("id", str(indice + 1))
        ET.SubElement(vehiculo, "Capacidad").text = str(300 * cantidad_productos)
        ET.SubElement(vehiculo, "VelocidadMedia").text = "50"
        ET.SubElement(vehiculo, "DesvEstVelocidad").text = "5"

    contenido = minidom.parseString(ET.tostring(raiz)).toprettyxml(indent="")

    os.makedirs(carpeta_salida, exist_ok=True)
    ruta = Path(carpeta_salida) / nombre_archivo
    with open(ruta, "w", encoding="utf-8") as archivo:
        archivo.write(contenido)
    return ruta


def generar_conjunto(
    configuraciones: Sequence[str],
    carpeta_salida: Path,
    con_costo_almacenamiento: bool,
    repeticiones: int = 10,
    semilla_base: int = 1,
) -> int:
    """Genera `repeticiones` instancias por configuración y devuelve cuántas escribió."""
    sufijo_nombre = "Almacenamiento" if con_costo_almacenamiento else ""
    total = len(configuraciones) * repeticiones
    print(f"Generando {total} instancias en {carpeta_salida} ({repeticiones} por configuración)...")

    for indice_configuracion, configuracion in enumerate(configuraciones):
        codigo_clientes, codigo_vehiculos, codigo_productos, codigo_geografia = (
            configuracion[0:2], configuracion[2:4], configuracion[4:6], configuracion[6:8],
        )
        for repeticion in range(repeticiones):
            semilla = semilla_base + (indice_configuracion * repeticiones) + repeticion
            random.seed(semilla)
            np.random.seed(semilla)

            minimo_clientes, maximo_clientes = RANGO_CLIENTES[codigo_clientes]
            clientes = random.randint(minimo_clientes, maximo_clientes)

            nombre_archivo = f"instancia_{configuracion}{sufijo_nombre}_rep{repeticion + 1}.xml"
            generar_instancia(
                ancho_zona=ANCHO_ZONA,
                largo_zona=LARGO_ZONA,
                horizonte_tiempo=HORIZONTE_TIEMPO,
                cantidad_productos=CANTIDAD_PRODUCTOS[codigo_productos],
                cantidad_clientes=clientes,
                cantidad_vehiculos_flota=cantidad_vehiculos(codigo_vehiculos, clientes),
                semilla=semilla,
                carpeta_salida=carpeta_salida,
                nombre_archivo=nombre_archivo,
                distribucion_geografica=codigo_geografia,
                con_costo_almacenamiento=con_costo_almacenamiento,
            )
            print(f"  {nombre_archivo} (semilla {semilla})")
    return total


def main() -> None:
    analizador = argparse.ArgumentParser(description="Genera los conjuntos de instancias del proyecto.")
    analizador.add_argument("--salida", type=Path, default=CARPETA_INSTANCIAS, help="Carpeta que contendrá los conjuntos.")
    argumentos = analizador.parse_args()

    generar_conjunto(
        CONFIGURACIONES_SIN_COSTO_ALMACENAMIENTO, argumentos.salida / "sin_costo_almacenamiento", con_costo_almacenamiento=False
    )
    generar_conjunto(
        CONFIGURACIONES_CON_COSTO_ALMACENAMIENTO, argumentos.salida / "con_costo_almacenamiento", con_costo_almacenamiento=True
    )
    print("Proceso completado.")


if __name__ == "__main__":
    main()
