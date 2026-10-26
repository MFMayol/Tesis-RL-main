# Simulador IRP

Proyecto para simular y comparar politicas de inventario y ruteo (IRP) con demanda y velocidad de transporte estocasticas.

## Requisitos y ejecucion

Se requiere Python 3.10 o posterior. Desde PowerShell, abre la carpeta del proyecto y prepara un entorno:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

Si PowerShell bloquea la activacion del entorno, ejecuta `python -m pip install -r requirements.txt` con el Python que vayas a usar y luego `python main.py`. Si `python` no se reconoce en Windows, prueba `py` en su lugar.

## Uso del menu

1. Elige una politica con su numero, varias separadas por coma (por ejemplo `1,6`) o escribe `todas`.
2. Para cada politica de aprendizaje, configura los hiperparametros. Pulsa Enter para aceptar el valor entre corchetes; escribe `d` para conservar los valores por defecto restantes de esa politica. Los hiperparametros invalidos se vuelven a preguntar.
3. Elige un conjunto bajo `datos/instancias/`, despues una configuracion y las repeticiones. Para usar otros XML, elige `m` e introduce una ruta, carpeta o patron; separa varias rutas con punto y coma. Por ejemplo: `datos/instancias/con_costo_almacenamiento/instancia_C2M2P2GUAlmacenamiento_rep1.xml`.
4. Revisa el resumen y confirma con Enter. Si ya hay un CSV para una politica e instancia, elige reemplazar, omitir o cancelar.

Escribe `q` en cualquier menu para cancelar. Las politicas de simulacion suelen tardar menos que las de aprendizaje: las entrenables usan 30 000 episodios por defecto, las heuristicas 1000 simulaciones y rollout 5 o 20. Puedes reducir episodios antes de una corrida de prueba.

## Politicas

Definidas en `simulador_irp/catalogo_politicas.py`:

- `heuristica_simple` y `heuristica_clusterizada`: atienden al cliente mas critico; la segunda agrupa clientes por vehiculo con k-means.
- `eoq_estocastico`: cantidad de entrega y punto de reorden (Q, R) segun la distancia real al cliente.
- `rollout_heuristica_simple` y `rollout_heuristica_clusterizada`: eligen la accion de menor costo esperado simulando una heuristica.
- `monte_carlo_lineal`, `monte_carlo_fourier`: valor Q lineal, con y sin base de Fourier.
- `monte_carlo_fourier_entregas_parciales` y `monte_carlo_fourier_criticidad`: entregas parciales; la segunda prioriza productos por penalizacion / costo de almacenamiento.
- `monte_carlo_red_neuronal` y `q_learning_red_neuronal`, cada una con variante `_entregas_parciales`.

Al elegir una politica de aprendizaje, el programa pregunta sus hiperparametros: episodios de entrenamiento, epsilon y tasa de aprendizaje, y segun la politica el orden de Fourier, las neuronas por capa, el factor de descuento, el tamano del lote y la capacidad de la memoria de experiencias. Enter conserva el valor por defecto (30 000 episodios) y `d` usa los valores por defecto del resto de la politica; cada valor se valida al ingresarlo. Los valores usados quedan en la columna `hiperparametros` del CSV. Las politicas de simulacion promedian 1000 simulaciones (5 y 20 en los rollout).

## Resultados

Cada politica guarda sus resultados en su propia carpeta: `resultados/<politica>/resultados.csv`, con una fila por instancia (funcion objetivo, desglose de costos, porcentajes, tamano de la instancia, tiempos e hiperparametros). Ejecutar de nuevo una politica sobre la misma instancia reemplaza su fila. Los CSV usan codificacion UTF-8 con BOM, por lo que Excel muestra bien los acentos. Si un CSV esta abierto en Excel, el programa avisa al guardar y ofrece reintentar o dejar una copia en `resultados/resultados_pendientes.csv`.

Las politicas entrenables guardan ademas su curva de entrenamiento en `resultados/<politica>/graficos/` (PNG y CSV por instancia).

## Estructura

- `main.py`: punto de entrada.
- `simulador_irp/`: codigo del simulador (`nucleo/`, `politicas/`, `utilidades/` e interfaz de consola).
- `datos/instancias/`: instancias XML (`sin_costo_almacenamiento/`, `con_costo_almacenamiento/`). El nombre `C2M2P1GU` codifica clientes, vehiculos, productos y geografia (uniforme o agrupada); el menu lo explica.
- `herramientas/generar_instancias.py`: regenera las instancias de forma reproducible.