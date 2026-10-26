from dataclasses import dataclass, field, replace
from enum import Enum
from importlib import import_module
from typing import Any, Mapping, Tuple


class TipoPolitica(Enum):
    SIMULACION = "simulación"  # Se evalúa promediando simulaciones de la política tal cual.
    APRENDIZAJE = "aprendizaje"  # Se entrena un modelo y se reporta el mejor resultado encontrado.


@dataclass(frozen=True)
class EspecificacionPolitica:
    """Describe una política ejecutable: qué clase usar, con qué parámetros y cómo evaluarla."""

    clave: str
    nombre: str
    familia: str
    descripcion: str
    clase: str  # "ruta.del.modulo:NombreClase"
    tipo: TipoPolitica
    parametros: Mapping[str, Any] = field(default_factory=dict)
    simulaciones: int = 0  # Solo para TipoPolitica.SIMULACION.
    umbral_inventario_clientes: float = 0.2
    umbral_inventario_vehiculos: float = 0.1

    @property
    def es_aprendizaje(self) -> bool:
        return self.tipo is TipoPolitica.APRENDIZAJE

    def con_parametros(self, parametros: Mapping[str, Any]) -> "EspecificacionPolitica":
        return replace(self, parametros=dict(parametros))

    def crear_politica(self, instancia, simulador):
        nombre_modulo, nombre_clase = self.clase.split(":")
        return getattr(import_module(nombre_modulo), nombre_clase)(instancia, simulador, **self.parametros)


_HEURISTICAS = "simulador_irp.politicas.heuristicas"
_APROXIMACION = "simulador_irp.politicas.aproximacion_funcion_valor"
_PARCIALES = "simulador_irp.politicas.entregas_parciales"
_REDES = "simulador_irp.politicas.redes_neuronales"

_FAMILIA_HEURISTICAS = "Heurísticas"
_FAMILIA_ROLLOUT = "Rollout"
_FAMILIA_LINEAL = "Monte Carlo lineal y Fourier"
_FAMILIA_REDES = "Redes neuronales"

POLITICAS: Tuple[EspecificacionPolitica, ...] = (
    EspecificacionPolitica(
        "heuristica_simple", "Heurística simple", _FAMILIA_HEURISTICAS,
        "Atiende al cliente más crítico con el primer vehículo disponible.",
        f"{_HEURISTICAS}:PoliticaHeuristicaSimple", TipoPolitica.SIMULACION,
        simulaciones=1000, umbral_inventario_vehiculos=0.3,
    ),
    EspecificacionPolitica(
        "heuristica_clusterizada", "Heurística clusterizada", _FAMILIA_HEURISTICAS,
        "Como la simple, pero cada vehículo atiende solo a los clientes de su grupo k-means.",
        f"{_HEURISTICAS}:PoliticaHeuristicaClusterizada", TipoPolitica.SIMULACION,
        simulaciones=1000, umbral_inventario_vehiculos=0.3,
    ),
    EspecificacionPolitica(
        "eoq_estocastico", "EOQ estocástico (Q, R)", _FAMILIA_HEURISTICAS,
        "Calcula la cantidad de entrega y el punto de reorden según la distancia real al cliente.",
        "simulador_irp.politicas.eoq_estocastico:PoliticaEOQEstocastico", TipoPolitica.SIMULACION,
        simulaciones=1000,
    ),
    EspecificacionPolitica(
        "rollout_heuristica_simple", "Rollout sobre heurística simple", _FAMILIA_ROLLOUT,
        "Elige la acción de menor costo esperado simulando la heurística simple.",
        "simulador_irp.politicas.rollout:PoliticaRolloutHeuristicaSimple", TipoPolitica.SIMULACION,
        simulaciones=5, umbral_inventario_vehiculos=0.3,
    ),
    EspecificacionPolitica(
        "rollout_heuristica_clusterizada", "Rollout sobre heurística clusterizada", _FAMILIA_ROLLOUT,
        "Elige la acción de menor costo esperado simulando la heurística clusterizada.",
        "simulador_irp.politicas.rollout:PoliticaRolloutHeuristicaClusterizada", TipoPolitica.SIMULACION,
        simulaciones=20, umbral_inventario_vehiculos=0.2,
    ),
    EspecificacionPolitica(
        "monte_carlo_lineal", "Monte Carlo lineal", _FAMILIA_LINEAL,
        "Aprende un valor Q lineal en las características del estado y la acción.",
        f"{_APROXIMACION}:MonteCarloLineal", TipoPolitica.APRENDIZAJE,
        parametros={"episodios": 30000, "epsilon": 0.1, "tasa_aprendizaje": 0.00001},
    ),
    EspecificacionPolitica(
        "monte_carlo_fourier", "Monte Carlo con base de Fourier", _FAMILIA_LINEAL,
        "Valor Q lineal sobre una expansión de Fourier de las características e interacciones.",
        f"{_APROXIMACION}:MonteCarloFourier", TipoPolitica.APRENDIZAJE,
        parametros={"episodios": 30000, "epsilon": 0.05, "tasa_aprendizaje": 0.00001, "orden_fourier": 5},
    ),
    EspecificacionPolitica(
        "monte_carlo_fourier_entregas_parciales", "Monte Carlo Fourier con entregas parciales", _FAMILIA_LINEAL,
        "Como Monte Carlo Fourier, evaluando además entregas parciales (útil con costo de almacenamiento).",
        f"{_PARCIALES}:MonteCarloFourierEntregasParciales", TipoPolitica.APRENDIZAJE,
        parametros={"episodios": 30000, "epsilon": 0.05, "tasa_aprendizaje": 0.0001, "orden_fourier": 5},
    ),
    EspecificacionPolitica(
        "monte_carlo_fourier_criticidad", "Monte Carlo Fourier con entregas por criticidad", _FAMILIA_LINEAL,
        "Como Monte Carlo Fourier, con entregas parciales que priorizan productos por penalización / almacenamiento.",
        f"{_PARCIALES}:MonteCarloFourierCriticidad", TipoPolitica.APRENDIZAJE,
        parametros={"episodios": 30000, "epsilon": 0.05, "tasa_aprendizaje": 0.0001, "orden_fourier": 5},
    ),
    EspecificacionPolitica(
        "monte_carlo_red_neuronal", "Monte Carlo con red neuronal", _FAMILIA_REDES,
        "Aprende el valor Q con una red neuronal a partir de episodios completos.",
        f"{_REDES}.monte_carlo_red_neuronal:MonteCarloRedNeuronal", TipoPolitica.APRENDIZAJE,
        parametros={"episodios": 30000, "epsilon": 0.05, "tasa_aprendizaje": 0.0001, "neuronas_por_capa_oculta": [64, 64, 64, 64]},
    ),
    EspecificacionPolitica(
        "monte_carlo_red_neuronal_entregas_parciales", "Monte Carlo con red neuronal y entregas parciales", _FAMILIA_REDES,
        "Como Monte Carlo con red neuronal, evaluando además entregas parciales.",
        f"{_REDES}.monte_carlo_red_neuronal:MonteCarloRedNeuronalEntregasParciales", TipoPolitica.APRENDIZAJE,
        parametros={"episodios": 30000, "epsilon": 0.05, "tasa_aprendizaje": 0.0001, "neuronas_por_capa_oculta": [128] * 8},
    ),
    EspecificacionPolitica(
        "q_learning_red_neuronal", "Q-learning con red neuronal", _FAMILIA_REDES,
        "Q-learning profundo (Double DQN) con memoria de experiencias.",
        f"{_REDES}.q_learning_red_neuronal:QLearningRedNeuronal", TipoPolitica.APRENDIZAJE,
        parametros={
            "episodios": 30000, "epsilon": 0.05, "tasa_aprendizaje": 0.001, "factor_descuento": 1,
            "tamano_lote": 32, "capacidad_memoria_experiencias": 20000, "neuronas_por_capa_oculta": [64, 64],
        },
    ),
    EspecificacionPolitica(
        "q_learning_red_neuronal_entregas_parciales", "Q-learning con red neuronal y entregas parciales", _FAMILIA_REDES,
        "Como Q-learning con red neuronal, evaluando además entregas parciales.",
        f"{_REDES}.q_learning_red_neuronal:QLearningRedNeuronalEntregasParciales", TipoPolitica.APRENDIZAJE,
        parametros={
            "episodios": 30000, "epsilon": 0.05, "tasa_aprendizaje": 0.001, "factor_descuento": 1,
            "tamano_lote": 32, "capacidad_memoria_experiencias": 20000, "neuronas_por_capa_oculta": [64, 64, 64, 64],
        },
    ),
)
