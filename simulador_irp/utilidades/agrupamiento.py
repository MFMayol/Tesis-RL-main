from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans


def agrupar_clientes_por_vehiculo(
    cantidad_vehiculos: int,
    posiciones_clientes: Dict[int, Dict[str, float]],
) -> Dict[int, List[int]]:
    """Agrupa los clientes con k-means según su posición.

    Args:
        cantidad_vehiculos: cantidad de grupos a formar.
        posiciones_clientes: {id_cliente: {'x': coordenada_x, 'y': coordenada_y}}.

    Returns:
        {id_vehiculo (desde 1): [id_cliente, ...]}.
    """
    ids_clientes = list(posiciones_clientes.keys())
    coordenadas = np.array(
        [[posiciones_clientes[id_cliente]['x'], posiciones_clientes[id_cliente]['y']] for id_cliente in ids_clientes]
    )
    modelo = KMeans(n_clusters=cantidad_vehiculos, init='k-means++', n_init=10, random_state=42)
    etiquetas = modelo.fit_predict(coordenadas)

    grupos = {id_vehiculo + 1: [] for id_vehiculo in range(cantidad_vehiculos)}
    for indice, etiqueta in enumerate(etiquetas):
        grupos[etiqueta + 1].append(ids_clientes[indice])
    return grupos
