import pandas as pd

from simulador_irp.rutas import carpeta_graficos_politica


def guardar_grafico_entrenamiento(clave_politica: str, nombre_instancia: str, serie: pd.DataFrame):
    """Guarda la curva de entrenamiento (PNG) y sus valores (CSV) en resultados/<política>/graficos/."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    carpeta = carpeta_graficos_politica(clave_politica)
    carpeta.mkdir(parents=True, exist_ok=True)
    serie.to_csv(carpeta / f"{nombre_instancia}.csv", index=False)

    figura, eje = plt.subplots(figsize=(8, 6))
    eje.plot(serie["episodios_entrenados"], serie["mejor_funcion_objetivo"], label="Best value found", linewidth=2)
    eje.plot(
        serie["episodios_entrenados"], serie["funcion_objetivo_evaluada"],
        label="Periodic evaluation", linestyle="--", linewidth=2,
    )
    eje.set_xlabel("Training episodes")
    eje.set_ylabel("Objective function (total cost)")
    eje.set_title(f"Training progress: {nombre_instancia}")
    eje.legend()
    eje.grid(True, alpha=0.3)
    figura.tight_layout()
    figura.savefig(carpeta / f"{nombre_instancia}.png", dpi=150)
    plt.close(figura)
