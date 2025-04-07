import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Union, Optional
import logging
from trigrs_calculations import (
    analyze_grid_cell,
    calculate_factor_of_safety,
    calculate_pressure_head
)
from visualization import generate_map, generate_time_series_plot

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TRIGRSModel:
    """
    Implementación optimizada del modelo TRIGRS (Transient Rainfall Infiltration and 
    Grid-Based Regional Slope-Stability Analysis).
    """
    def __init__(self, project_name: str, workspace_dir: Optional[str] = None):
        """
        Inicializa el modelo TRIGRS.
        
        Args:
            project_name: Nombre del proyecto
            workspace_dir: Directorio de trabajo (opcional, por defecto el directorio actual)
        """
        self.project_name = project_name
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
        
        # Crear estructura de directorios
        self.dirs = self._create_directory_structure()
        
        # Inicializar parámetros por defecto
        self.parameters = {
            'flow_direction_scheme': 2,  # ESRI=1, TopoIndex=2
            'weight_exponent': 25,
            'iterations': 10,
            'min_slope_angle': 0.0,
            'max_slope_angle': 90.0,
            'time_steps': 48,  # Número de pasos de tiempo
            'rainfall_intensity': None,  # Se debe establecer después
            'dem_file': None,  # Se debe establecer después
            'slope_file': None,  # Se debe establecer después
            'zones_file': None,  # Se debe establecer después
            'directions_file': None,  # Se debe establecer después
            'zmax_file': None,  # Se debe establecer después
            'depthwt_file': None,  # Se debe establecer después
        }
        
        # Parámetros geotécnicos por zona
        self.geotechnical_params = {}
        
        # Datos de entrada
        self.dem_data = None
        self.slope_data = None
        self.zones_data = None
        self.directions_data = None
        self.zmax_data = None
        self.depthwt_data = None
        
        # Resultados
        self.results = None
        
        logger.info(f"Modelo TRIGRS inicializado para el proyecto: {project_name}")
        logger.info(f"Directorio de trabajo: {self.workspace_dir}")

    def _create_directory_structure(self) -> Dict[str, Path]:
        """Crea la estructura de directorios necesaria para el modelo."""
        dirs = {
            'root': self.workspace_dir / 'TRIGRS',
            'inputs': self.workspace_dir / 'TRIGRS' / 'inputs',
            'outputs': self.workspace_dir / 'TRIGRS' / 'outputs',
            'temp': self.workspace_dir / 'TRIGRS' / 'temp'
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directorio creado/verificado: {dir_path}")
            
        return dirs

    def _load_ascii_grid(self, file_path: Path) -> np.ndarray:
        """Carga un archivo ASCII grid y retorna los datos como array."""
        try:
            # Leer encabezado
            with open(file_path, 'r') as f:
                header = {}
                for _ in range(6):
                    key, value = f.readline().strip().split()
                    header[key.lower()] = float(value)
            
            # Leer datos
            data = np.loadtxt(file_path, skiprows=6)
            
            # Reemplazar valores NODATA con np.nan solo si están exactamente en el valor NODATA
            nodata_value = header.get('nodata_value', -9999)
            data = np.where(np.isclose(data, nodata_value, rtol=1e-5), np.nan, data)
            
            # Imprimir información de diagnóstico
            print(f"\nDiagnóstico para {file_path}:")
            print(f"- Dimensiones: {data.shape}")
            print(f"- Valor mínimo: {np.nanmin(data)}")
            print(f"- Valor máximo: {np.nanmax(data)}")
            print(f"- Número de NaN: {np.isnan(data).sum()}")
            print(f"- Valor NODATA del encabezado: {nodata_value}")
            
            return data, header
            
        except Exception as e:
            logger.error(f"Error al cargar archivo {file_path}: {e}")
            raise

    def set_input_files(self, 
                       dem_file: str, 
                       slope_file: str, 
                       zones_file: str,
                       directions_file: str,
                       zmax_file: str,
                       depthwt_file: str) -> None:
        """
        Establece y carga los archivos de entrada necesarios.
        
        Args:
            dem_file: Ruta al archivo DEM (Digital Elevation Model)
            slope_file: Ruta al archivo de pendientes
            zones_file: Ruta al archivo de zonas
            directions_file: Ruta al archivo de direcciones de flujo
            zmax_file: Ruta al archivo de profundidad máxima del suelo
            depthwt_file: Ruta al archivo de profundidad del nivel freático
        """
        try:
            # Actualizar rutas
            self.parameters.update({
                'dem_file': Path(dem_file),
                'slope_file': Path(slope_file),
                'zones_file': Path(zones_file),
                'directions_file': Path(directions_file),
                'zmax_file': Path(zmax_file),
                'depthwt_file': Path(depthwt_file)
            })
            
            # Cargar datos
            self.dem_data, self.dem_header = self._load_ascii_grid(dem_file)
            self.slope_data, _ = self._load_ascii_grid(slope_file)
            self.zones_data, _ = self._load_ascii_grid(zones_file)
            self.directions_data, _ = self._load_ascii_grid(directions_file)
            self.zmax_data, _ = self._load_ascii_grid(zmax_file)
            self.depthwt_data, _ = self._load_ascii_grid(depthwt_file)
            
            # Verificar dimensiones
            shapes = [
                self.dem_data.shape,
                self.slope_data.shape,
                self.zones_data.shape,
                self.directions_data.shape,
                self.zmax_data.shape,
                self.depthwt_data.shape
            ]
            if not all(shape == shapes[0] for shape in shapes):
                raise ValueError("Las dimensiones de los archivos de entrada no coinciden")
                
            logger.info("Archivos de entrada cargados correctamente")
            
        except Exception as e:
            logger.error(f"Error al cargar archivos de entrada: {e}")
            raise

    def set_geotechnical_parameters(self, zone_id: int, params: Dict[str, float]) -> None:
        """
        Establece los parámetros geotécnicos para una zona específica.
        
        Args:
            zone_id: ID de la zona
            params: Diccionario con los parámetros geotécnicos
        """
        required_params = [
            'c', 'phi', 'gamma_sat', 'D_sat', 
            'K_sat', 'theta_sat', 'theta_res', 'alpha'
        ]
        
        # Verificar parámetros
        missing_params = [p for p in required_params if p not in params]
        if missing_params:
            raise ValueError(f"Faltan los siguientes parámetros: {missing_params}")
            
        self.geotechnical_params[zone_id] = params
        logger.info(f"Parámetros geotécnicos establecidos para la zona {zone_id}")

    def set_rainfall_intensity(self, intensities: List[float], durations: List[float]) -> None:
        """
        Establece la intensidad de lluvia para diferentes períodos.
        
        Args:
            intensities: Lista de intensidades de lluvia (mm/h)
            durations: Lista de duraciones correspondientes (h)
        """
        if len(intensities) != len(durations):
            raise ValueError("Las listas de intensidades y duraciones deben tener la misma longitud")
            
        # Convertir mm/h a m/s
        intensities_ms = np.array(intensities) * (1/1000) * (1/3600)
        durations_s = np.array(durations) * 3600
            
        self.parameters['rainfall_intensity'] = {
            'intensities': intensities_ms,
            'durations': durations_s
        }
        logger.info("Intensidades de lluvia establecidas correctamente")

    def run_analysis(self) -> Dict:
        """
        Ejecuta el análisis TRIGRS completo.
        
        Returns:
            Dict: Resultados del análisis
        """
        try:
            # Verificar datos de entrada
            self._verify_input_data()
            
            # Obtener dimensiones y tiempos
            shape = self.dem_data.shape
            time_steps = self.parameters['time_steps']
            n_times = len(time_steps)
            
            # Inicializar arrays de resultados con NaN
            self.results = {
                'factor_of_safety': np.full((shape[0], shape[1], n_times), np.nan),
                'pressure_head': np.full((shape[0], shape[1], n_times), np.nan),
                'infiltration': np.full((shape[0], shape[1], n_times), np.nan),
                'runoff': np.full((shape[0], shape[1], n_times), np.nan)
            }
            
            # Contadores para diagnóstico
            total_cells = shape[0] * shape[1]
            processed_cells = 0
            skipped_cells = 0
            nan_dem = 0
            nan_slope = 0
            nan_zones = 0
            nan_directions = 0
            nan_zmax = 0
            nan_depthwt = 0
            
            # Iterar sobre cada celda válida
            for i in range(shape[0]):
                for j in range(shape[1]):
                    # Verificar cada valor individualmente y contar NaN
                    if np.isnan(self.dem_data[i,j]):
                        nan_dem += 1
                        continue
                    if np.isnan(self.slope_data[i,j]):
                        nan_slope += 1
                        continue
                    if np.isnan(self.zones_data[i,j]):
                        nan_zones += 1
                        continue
                    if np.isnan(self.directions_data[i,j]):
                        nan_directions += 1
                        continue
                    if np.isnan(self.zmax_data[i,j]):
                        nan_zmax += 1
                        continue
                    if np.isnan(self.depthwt_data[i,j]):
                        nan_depthwt += 1
                        continue
                    
                    try:
                        # Obtener zona y parámetros
                        zone = int(self.zones_data[i,j])
                        if zone not in self.geotechnical_params:
                            skipped_cells += 1
                            continue
                        
                        # Analizar celda usando todas las variables
                        results = analyze_grid_cell(
                            elevation=self.dem_data[i,j],
                            slope=self.slope_data[i,j],
                            soil_params=self.geotechnical_params[zone],
                            rainfall_data=self.parameters['rainfall_intensity'],
                            initial_conditions={
                                'depth': self.zmax_data[i,j],
                                'water_table_depth': self.depthwt_data[i,j],
                                'flow_direction': self.directions_data[i,j]
                            },
                            time_steps=time_steps
                        )
                        
                        # Almacenar resultados
                        for key in self.results:
                            self.results[key][i,j,:] = results[key]
                        
                        processed_cells += 1
                        
                    except Exception as e:
                        logger.warning(f"Error al procesar celda ({i},{j}): {str(e)}")
                        skipped_cells += 1
            
            # Imprimir resumen de diagnóstico
            print("\nResumen del análisis:")
            print(f"Total de celdas: {total_cells}")
            print(f"Celdas procesadas: {processed_cells}")
            print(f"Celdas omitidas: {skipped_cells}")
            print(f"Celdas con DEM NaN: {nan_dem}")
            print(f"Celdas con pendiente NaN: {nan_slope}")
            print(f"Celdas con zona NaN: {nan_zones}")
            print(f"Celdas con dirección NaN: {nan_directions}")
            print(f"Celdas con profundidad NaN: {nan_zmax}")
            print(f"Celdas con nivel freático NaN: {nan_depthwt}")
            
            if processed_cells > 0:
                print("Análisis completado exitosamente")
                return {'status': 'success', 'message': 'Análisis completado'}
            else:
                raise ValueError("No se procesó ninguna celda")
            
        except Exception as e:
            logger.error(f"Error durante el análisis: {str(e)}")
            raise

    def _verify_input_data(self) -> None:
        """Verifica que todos los datos necesarios estén presentes."""
        if any(v is None for v in [
            self.dem_data,
            self.slope_data,
            self.zones_data,
            self.directions_data,
            self.zmax_data,
            self.depthwt_data
        ]):
            raise ValueError("No se han establecido todos los archivos de entrada necesarios")
            
        if not self.geotechnical_params:
            raise ValueError("No se han establecido parámetros geotécnicos para ninguna zona")
            
        if self.parameters['rainfall_intensity'] is None:
            raise ValueError("No se han establecido los datos de intensidad de lluvia")

    def plot_results(self, result_type: str, time_step: int = -1) -> None:
        """
        Genera un mapa de resultados para un tipo y tiempo específico.
        
        Args:
            result_type: Tipo de resultado ('factor_of_safety', 'pressure_head', 'infiltration', 'runoff')
            time_step: Paso de tiempo a graficar (-1 para el último)
        """
        try:
            # Verificar tipo de resultado
            valid_types = ['factor_of_safety', 'pressure_head', 'infiltration', 'runoff']
            if result_type not in valid_types:
                raise ValueError(f"Tipo de resultado no válido. Debe ser uno de: {valid_types}")
            
            # Obtener datos para el tiempo especificado
            if time_step == -1:
                time_step = self.results[result_type].shape[2] - 1
            
            data = self.results[result_type][:,:,time_step]
            
            # Configurar límites y colores según el tipo de resultado
            if result_type == 'factor_of_safety':
                vmin, vmax = 0, 2
                cmap = 'RdYlGn'  # Rojo para valores bajos (inseguro), verde para valores altos (seguro)
            elif result_type == 'pressure_head':
                vmin, vmax = -10, 10
                cmap = 'RdBu_r'  # Rojo para valores positivos (presión), azul para negativos (succión)
            elif result_type == 'infiltration':
                vmin, vmax = 0, np.nanmax(data)
                cmap = 'Blues'  # Azules más intensos para mayor infiltración
            else:  # runoff
                vmin, vmax = 0, np.nanmax(data)
                cmap = 'Blues'  # Azules más intensos para mayor escorrentía
            
            # Crear figura
            plt.figure(figsize=(10, 8))
            
            # Generar mapa
            im = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, label=result_type.replace('_', ' ').title())
            
            # Configurar título y etiquetas
            plt.title(f"{result_type.replace('_', ' ').title()} - Time step {time_step}")
            plt.xlabel('X (cells)')
            plt.ylabel('Y (cells)')
            
            # Guardar figura
            output_dir = self.dirs['outputs'] / 'maps'
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"{result_type}_t{time_step}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Mapa guardado en: {output_file}")
            
        except Exception as e:
            logger.error(f"Error al generar mapa de resultados: {str(e)}")
            raise

    def export_results(self, output_format: str = 'ascii') -> None:
        """
        Exporta los resultados del análisis.
        
        Args:
            output_format: Formato de salida ('ascii', 'csv')
        """
        if self.results is None:
            raise ValueError("No hay resultados para exportar. Ejecute el análisis primero.")
            
        try:
            for result_type, data in self.results.items():
                # Para cada paso de tiempo
                for t in range(data.shape[2]):
                    if output_format == 'ascii':
                        # Crear archivo ASCII grid
                        output_file = self.dirs['outputs'] / f"{result_type}_t{t}.asc"
                        with open(output_file, 'w') as f:
                            # Escribir encabezado
                            for key, value in self.dem_header.items():
                                f.write(f"{key.upper()} {value}\n")
                            # Escribir datos
                            np.savetxt(f, data[:,:,t], fmt='%.6f')
                    elif output_format == 'csv':
                        # Exportar como CSV
                        output_file = self.dirs['outputs'] / f"{result_type}_t{t}.csv"
                        pd.DataFrame(data[:,:,t]).to_csv(output_file, index=False)
                    else:
                        raise ValueError(f"Formato de salida no soportado: {output_format}")
                        
            logger.info(f"Resultados exportados en formato {output_format}")
            
        except Exception as e:
            logger.error(f"Error al exportar resultados: {e}")
            raise

    def generate_result_maps(self):
        """
        Genera mapas para todos los tiempos de análisis.
        """
        try:
            # Cargar DEM para hillshade
            dem = self.dem_data
            
            # Obtener tiempos de análisis
            time_steps = np.array(self.parameters['time_steps'])
            
            # Generar mapas para cada paso de tiempo
            for i, t in enumerate(time_steps):
                hours = t/3600  # Convertir segundos a horas
                
                # Factor de Seguridad
                fs_data = self.results['factor_of_safety'][:, :, i]
                output_path = self.dirs['outputs'] / 'maps' / f'factor_of_safety_t{i}.png'
                title = f'Factor de Seguridad - t = {hours:.1f} h'
                generate_map(fs_data, output_path, title, dem=dem, result_type='factor_of_safety')
                
                # Presión de poros
                pressure_data = self.results['pressure_head'][:, :, i]
                output_path = self.dirs['outputs'] / 'maps' / f'pressure_head_t{i}.png'
                title = f'Presión de Poros - t = {hours:.1f} h'
                generate_map(pressure_data, output_path, title, dem=dem, result_type='pressure_head')
                
                # Infiltración
                infiltration_data = self.results['infiltration'][:, :, i]
                output_path = self.dirs['outputs'] / 'maps' / f'infiltration_t{i}.png'
                title = f'Infiltración - t = {hours:.1f} h'
                generate_map(infiltration_data, output_path, title, dem=dem, result_type='infiltration')
                
                # Escorrentía
                runoff_data = self.results['runoff'][:, :, i]
                output_path = self.dirs['outputs'] / 'maps' / f'runoff_t{i}.png'
                title = f'Escorrentía - t = {hours:.1f} h'
                generate_map(runoff_data, output_path, title, dem=dem, result_type='runoff')
            
            # Generar gráfico de series temporales
            output_path = self.dirs['outputs'] / 'maps' / 'time_series.png'
            generate_time_series_plot(self.results, str(output_path), time_steps)
            
            logger.info("Mapas y gráficos generados exitosamente")
            
        except Exception as e:
            logger.error(f"Error al generar mapas: {str(e)}")
            raise

    def load_ascii_grid(self, filepath: str) -> np.ndarray:
        """
        Carga un archivo ASCII grid y retorna el array de datos.
        
        Args:
            filepath: Ruta al archivo ASCII
        
        Returns:
            np.ndarray: Array con los datos del grid
        """
        try:
            with open(filepath, 'r') as f:
                # Leer encabezado
                ncols = int(f.readline().split()[1])
                nrows = int(f.readline().split()[1])
                xllcorner = float(f.readline().split()[1])
                yllcorner = float(f.readline().split()[1])
                cellsize = float(f.readline().split()[1])
                nodata = float(f.readline().split()[1])
                
                # Leer datos
                data = np.loadtxt(f)
                
                # Reemplazar valores NODATA por NaN
                data[data == nodata] = np.nan
                
                return data
                
        except Exception as e:
            logger.error(f"Error al cargar archivo ASCII grid {filepath}: {str(e)}")
            raise 