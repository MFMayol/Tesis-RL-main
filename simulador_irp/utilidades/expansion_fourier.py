import numpy as np


def expansion_fourier_unidimensional(caracteristicas, orden=5):
    """Expande cada característica con cos(pi*i*x) y sin(pi*i*x), para i = 1..orden.

    Args:
        caracteristicas: arreglo de forma (n,), normalizado.
        orden: orden de la aproximación.

    Returns:
        Arreglo de forma (n * 2 * orden,), agrupado por característica.
    """
    indices = np.arange(1, orden + 1).reshape(-1, 1)
    argumento = np.pi * indices * caracteristicas.reshape(1, -1)
    cosenos = np.cos(argumento)
    senos = np.sin(argumento)
    # El orden 'F' deja juntos los términos de una misma característica.
    return np.vstack([cosenos, senos]).reshape(-1, order='F')


def expansion_fourier_interacciones(caracteristica_1, caracteristica_2, orden_1=5, orden_2=5):
    """Expande pares de variables con cos(pi*(i*x + j*y)) y sin(pi*(i*x + j*y)).

    Args:
        caracteristica_1: valores de la primera variable (x), normalizados.
        caracteristica_2: valores de la segunda variable (y), normalizados.
        orden_1: orden para la primera variable.
        orden_2: orden para la segunda variable.

    Returns:
        Arreglo de forma (n_pares * 2 * orden_1 * orden_2,), agrupado por par.
    """
    valores_1 = np.asarray(caracteristica_1, dtype=np.float32).reshape(1, -1)
    valores_2 = np.asarray(caracteristica_2, dtype=np.float32).reshape(1, -1)

    malla_i, malla_j = np.meshgrid(np.arange(1, orden_1 + 1), np.arange(1, orden_2 + 1), indexing='ij')
    malla_i = malla_i.flatten().reshape(-1, 1)
    malla_j = malla_j.flatten().reshape(-1, 1)

    argumento = np.pi * (malla_i * valores_1 + malla_j * valores_2)

    intercalado = np.empty((2 * len(malla_i), valores_1.shape[1]), dtype=np.float32)
    intercalado[0::2, :] = np.cos(argumento)
    intercalado[1::2, :] = np.sin(argumento)
    return intercalado.T.flatten()
