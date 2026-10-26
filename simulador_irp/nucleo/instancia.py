from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
import xml.etree.ElementTree as ET


@dataclass
class Producto:
    id: int
    peso: float

    def __post_init__(self):
        if self.peso <= 0:
            raise ValueError(f"El peso del producto {self.id} debe ser positivo")


@dataclass
class Vehiculo:
    id: int
    capacidad: float
    velocidad_media: float
    desviacion_velocidad: float
    inventario: Dict[int, float] = field(default_factory=dict)


@dataclass
class Cliente:
    id: int
    posicion_x: float
    posicion_y: float
    capacidad_almacenamiento: float
    inventarios_iniciales: Dict[int, float]
    costos_inventario: Dict[int, float]
    costos_penalizacion: Dict[int, float]
    demanda_media: Dict[int, float]
    demanda_desviacion: Dict[int, float]

    def __post_init__(self):
        if self.capacidad_almacenamiento <= 0:
            raise ValueError(f"La capacidad de almacenamiento del cliente {self.id} debe ser positiva")

        productos = set(self.inventarios_iniciales.keys())
        datos_por_producto = (
            self.costos_inventario,
            self.costos_penalizacion,
            self.demanda_media,
            self.demanda_desviacion,
        )
        if not all(set(datos.keys()) == productos for datos in datos_por_producto):
            raise ValueError(f"Todos los diccionarios del cliente {self.id} deben tener los mismos productos")


class Instancia:
    """Instancia del problema IRP (ruteo e inventario) leída desde un archivo XML.

    Atributos:
        ruta_archivo: ruta al XML de la instancia.
        ancho_zona, largo_zona: dimensiones de la zona de distribución.
        deposito_x, deposito_y: coordenadas del depósito.
        horizonte_tiempo: duración de la simulación en periodos.
        productos, clientes, vehiculos: diccionarios indexados por id.
        demandas_medias: demanda media de cada producto promediada entre clientes.
        umbral_inventario_clientes: fracción de la capacidad bajo la cual un cliente genera un nuevo estado.
        umbral_inventario_vehiculos: fracción de la capacidad bajo la cual un vehículo debe volver al depósito.
    """

    def __init__(self, ruta_archivo, umbral_inventario_clientes: float, umbral_inventario_vehiculos: float):
        self.ruta_archivo = Path(ruta_archivo)
        if not self.ruta_archivo.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {ruta_archivo}")

        self.ancho_zona: float = 0
        self.largo_zona: float = 0
        self.deposito_x: float = 0
        self.deposito_y: float = 0
        self.horizonte_tiempo: int = 0
        self.productos: Dict[int, Producto] = {}
        self.clientes: Dict[int, Cliente] = {}
        self.vehiculos: Dict[int, Vehiculo] = {}
        self.demandas_medias: Optional[Dict[int, float]] = None
        self.id_clientes = []
        self.id_vehiculos = []
        self.id_productos = []
        self.umbral_inventario_clientes = umbral_inventario_clientes
        self.umbral_inventario_vehiculos = umbral_inventario_vehiculos

        self._cargar_desde_xml()
        self._inicializar_inventarios_vehiculos()
        self.id_clientes = list(self.clientes.keys())
        self.id_vehiculos = list(self.vehiculos.keys())
        self.id_productos = list(self.productos.keys())

    @property
    def nombre(self) -> str:
        return self.ruta_archivo.stem

    def _cargar_desde_xml(self) -> None:
        try:
            raiz = ET.parse(self.ruta_archivo).getroot()

            informacion_general = raiz.find('InformacionGeneral')
            if informacion_general is None:
                raise ValueError("No se encontró la sección InformacionGeneral")

            self.ancho_zona = float(informacion_general.find('AnchoZona').text)
            self.largo_zona = float(informacion_general.find('LargoZona').text)
            self.deposito_x = float(informacion_general.find('Depot_x').text)
            self.deposito_y = float(informacion_general.find('Depot_y').text)
            self.horizonte_tiempo = int(informacion_general.find('HorizonteTiempo').text)

            if any(dimension <= 0 for dimension in (self.ancho_zona, self.largo_zona, self.horizonte_tiempo)):
                raise ValueError("Las dimensiones y el horizonte de tiempo deben ser positivos")

            self._cargar_productos(raiz)
            self._cargar_vehiculos(raiz)
            self._cargar_clientes(raiz)
        except ET.ParseError as error:
            raise ET.ParseError(f"Error al parsear el archivo XML: {error}")
        except (ValueError, TypeError) as error:
            raise ValueError(f"Error en los datos del archivo XML: {error}")

    def _cargar_productos(self, raiz: ET.Element) -> None:
        seccion = raiz.find('Productos')
        if seccion is None:
            raise ValueError("No se encontró la sección Productos")

        for elemento in seccion.findall('Producto'):
            id_producto = int(elemento.get('id'))
            self.productos[id_producto] = Producto(id=id_producto, peso=float(elemento.find('Peso').text))

    def _cargar_vehiculos(self, raiz: ET.Element) -> None:
        seccion = raiz.find('Vehiculos')
        if seccion is None:
            raise ValueError("No se encontró la sección Vehiculos")

        for elemento in seccion.findall('Vehiculo'):
            id_vehiculo = int(elemento.get('id'))
            self.vehiculos[id_vehiculo] = Vehiculo(
                id=id_vehiculo,
                capacidad=float(elemento.find('Capacidad').text),
                velocidad_media=float(elemento.find('VelocidadMedia').text),
                desviacion_velocidad=float(elemento.find('DesvEstVelocidad').text),
            )

    def _cargar_clientes(self, raiz: ET.Element) -> None:
        seccion = raiz.find('Clientes')
        if seccion is None:
            raise ValueError("No se encontró la sección Clientes")

        for elemento in seccion.findall('Cliente'):
            id_cliente = int(elemento.get('id'))
            self.clientes[id_cliente] = Cliente(
                id=id_cliente,
                posicion_x=float(elemento.find('PosicionX').text),
                posicion_y=float(elemento.find('PosicionY').text),
                capacidad_almacenamiento=float(elemento.find('CapacidadAlmacenamiento').text),
                inventarios_iniciales=self._leer_valores_por_producto(elemento, 'Inventarios'),
                costos_inventario=self._leer_valores_por_producto(elemento, 'Costos', 'CostoInventario'),
                costos_penalizacion=self._leer_valores_por_producto(elemento, 'Costos', 'CostoPenalizacion'),
                demanda_media=self._leer_valores_por_producto(elemento, 'Demandas', 'Media'),
                demanda_desviacion=self._leer_valores_por_producto(elemento, 'Demandas', 'DesvEst'),
            )

    def _leer_valores_por_producto(
        self, elemento: ET.Element, seccion: str, subseccion: Optional[str] = None
    ) -> Dict[int, float]:
        """Lee un valor por producto dentro de una sección del cliente (p. ej. Costos/ProductoN/CostoInventario)."""
        elemento_seccion = elemento.find(seccion)
        if elemento_seccion is None:
            raise ValueError(f"No se encontró la sección {seccion}")

        valores = {}
        for id_producto in self.productos:
            ruta = f'Producto{id_producto}/{subseccion}' if subseccion else f'Producto{id_producto}'
            elemento_valor = elemento_seccion.find(ruta)
            if elemento_valor is None:
                raise ValueError(f"Faltan datos para el producto {id_producto}")
            valores[id_producto] = float(elemento_valor.text)
        return valores

    def _inicializar_inventarios_vehiculos(self) -> None:
        """Carga cada vehículo repartiendo su capacidad según la demanda media relativa de los productos."""
        self.demandas_medias = {}
        for id_producto in self.productos:
            demandas = [cliente.demanda_media[id_producto] for cliente in self.clientes.values()]
            self.demandas_medias[id_producto] = sum(demandas) / len(demandas)

        if not self.productos:
            return

        id_producto_base = list(self.productos.keys())[0]
        demanda_base = self.demandas_medias[id_producto_base]
        if demanda_base == 0:
            demanda_base = 1e-5  # Evita dividir por cero cuando la demanda base es nula.

        for vehiculo in self.vehiculos.values():
            denominador = sum(
                (self.demandas_medias[id_producto] / demanda_base) * self.productos[id_producto].peso
                for id_producto in self.productos
            )
            vehiculo.inventario[id_producto_base] = int(vehiculo.capacidad / denominador)

            for id_producto in self.productos:
                if id_producto != id_producto_base:
                    vehiculo.inventario[id_producto] = int(
                        (vehiculo.inventario[id_producto_base] / demanda_base) * self.demandas_medias[id_producto]
                    )
