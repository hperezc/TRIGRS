# TRIGRS - Transient Rainfall Infiltration and Grid-Based Regional Slope-Stability Analysis

## Descripción
TRIGRS (Transient Rainfall Infiltration and Grid-Based Regional Slope-Stability Analysis) es un modelo numérico que calcula los cambios temporales y espaciales en el factor de seguridad de taludes y la presión de poros debido a la infiltración de lluvia. El modelo combina:

- Un modelo de infiltración transitoria
- Análisis de estabilidad de taludes infinitos
- Análisis espacialmente distribuido basado en SIG

## Características Principales
- 🌧️ Modelación de infiltración de lluvia variable en el tiempo
- 📊 Cálculo del factor de seguridad distribuido espacialmente
- 🗺️ Integración con datos espaciales (DEM, mapas de suelos)
- 📈 Generación de series temporales de resultados
- 🔍 Herramientas de visualización y análisis

## Requisitos del Sistema
- Python 3.8 o superior
- Dependencias:
  - NumPy
  - Pandas
  - Matplotlib
  - GDAL
  - SciPy

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/hperezc/TRIGRS.git
cd TRIGRS
```

2. Crear un ambiente virtual (opcional pero recomendado):
```bash
python -m venv env
source env/bin/activate  # En Linux/Mac
env\\Scripts\\activate   # En Windows
```

3. Instalar dependencias:
```bash
pip install numpy pandas matplotlib gdal scipy
```

## Estructura del Proyecto
```
TRIGRS/
├── docs/               # Documentación detallada
├── inputs/            # Archivos de entrada
│   ├── dem.asc       # Modelo digital de elevación
│   ├── slope.asc     # Mapa de pendientes
│   └── zones.asc     # Zonificación geotécnica
├── outputs/          # Resultados y visualizaciones
├── scripts/          # Scripts del modelo
│   ├── trigrs_model.py   # Modelo principal
│   ├── trigrs_io.py      # Manejo de entrada/salida
│   └── visualization.py   # Visualización de resultados
└── README.md         # Este archivo
```

## Uso Básico

1. Preparar archivos de entrada en formato ASCII grid:
   - DEM (Modelo Digital de Elevación)
   - Mapa de pendientes
   - Mapa de zonas geotécnicas
   - Parámetros del suelo por zonas

2. Ejecutar el modelo:
```python
from scripts.trigrs_model import TRIGRS
model = TRIGRS()
model.run()
```

3. Visualizar resultados:
```python
from scripts.visualization import plot_results
plot_results()
```

## Ejemplos de Resultados
El modelo genera:
- Mapas de factor de seguridad para diferentes tiempos
- Mapas de presión de poros
- Mapas de infiltración
- Series temporales de variables clave
- Análisis de sensibilidad
- Validación de resultados

## Documentación
Para más detalles sobre:
- Teoría del modelo
- Guías de usuario
- Ejemplos detallados
- Validación y verificación

Consulte la carpeta `docs/` o el archivo `modelo_TRIGRS.pdf`

## Contribuciones
Las contribuciones son bienvenidas. Por favor:
1. Fork el repositorio
2. Cree una rama para su característica
3. Envíe un Pull Request

## Soporte
Para preguntas y soporte:
- Abra un issue en GitHub
- Contacte a los desarrolladores

## Citación
Si usa este modelo en su investigación, por favor cite:

```bibtex
@software{TRIGRS2024,
  author = {Pérez, Héctor C.},
  title = {TRIGRS - Transient Rainfall Infiltration and Grid-Based Regional Slope-Stability Analysis},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/hperezc/TRIGRS}
}
```

## Licencia
Este proyecto está bajo la Licencia MIT - vea el archivo [LICENSE](LICENSE) para detalles. 