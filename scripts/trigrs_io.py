import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_input_data() -> Dict[str, np.ndarray]:
    """
    Carga los archivos de entrada del modelo TRIGRS.
    
    Returns:
        Dict[str, np.ndarray]: Diccionario con los arrays de entrada
    """
    try:
        # Obtener directorio actual
        directorio_actual = Path.cwd()
        
        # Definir rutas a los archivos
        archivos_entrada = {
            'dem': directorio_actual / "inputs" / "dem_proc.asc",
            'slope': directorio_actual / "inputs" / "slope_proc.asc",
            'zones': directorio_actual / "inputs" / "zones_proc.asc",
            'flow_direction': directorio_actual / "inputs" / "directions.asc",
            'depth': directorio_actual / "inputs" / "zmax.asc",
            'water_table': directorio_actual / "inputs" / "depthwt.asc"
        }
        
        # Cargar archivos
        data = {}
        for nombre, ruta in archivos_entrada.items():
            if not ruta.exists():
                raise FileNotFoundError(f"No se encontró el archivo: {ruta}")
                
            # Cargar archivo ASCII
            with open(ruta, 'r') as f:
                # Leer encabezado
                ncols = int(f.readline().split()[1])
                nrows = int(f.readline().split()[1])
                xllcorner = float(f.readline().split()[1])
                yllcorner = float(f.readline().split()[1])
                cellsize = float(f.readline().split()[1])
                nodata = float(f.readline().split()[1])
                
                # Leer datos
                array = np.loadtxt(f)
                
                # Reemplazar NODATA por NaN
                array[array == nodata] = np.nan
                
                # Diagnósticos adicionales para Zmax
                if nombre == 'depth':
                    print("\nDiagnóstico de Zmax:")
                    print(f"Valor mínimo: {np.nanmin(array)}")
                    print(f"Valor máximo: {np.nanmax(array)}")
                    print(f"Valores únicos: {np.unique(array[~np.isnan(array)])}")
                    print(f"Número de celdas con valor 0: {np.sum(array == 0)}")
                    print(f"Número de celdas con valor 1: {np.sum(array == 1)}")
                    print(f"Número de celdas con NaN: {np.sum(np.isnan(array))}")
                
                # Si es el nivel freático, forzar a 50m
                if nombre == 'water_table':
                    array = np.full_like(array, 50.0)
                    array[np.isnan(array)] = np.nan
                
                data[nombre] = array
                
                logger.info(f"Archivo {nombre} cargado: {array.shape}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error al cargar archivos de entrada: {str(e)}")
        raise

def load_soil_params() -> Dict[int, Dict[str, float]]:
    """
    Carga los parámetros geotécnicos del suelo.
    
    Returns:
        Dict[int, Dict[str, float]]: Diccionario con parámetros por zona
    """
    try:
        # Zona 1: Roca alterada – Nivel IIB (Pgsc – ra)
        # Arenisca mal gradada con grava (moderadamente meteorizada)
        params_zona1 = {
            'c': 5.0,           # Cohesión (kPa)
            'phi': 35.0,        # Ángulo de fricción (grados)
            'gamma_sat': 19.0,  # Peso unitario saturado (kN/m³)
            'D_sat': 1e-5,      # Difusividad hidráulica saturada (m²/s)
            'K_sat': 5e-5,      # Conductividad hidráulica saturada (m/s)
            'theta_sat': 0.45,  # Contenido volumétrico de agua saturada
            'theta_res': 0.05,  # Contenido volumétrico de agua residual
            'alpha': 0.02       # Parámetro de van Genuchten (1/m)
        }
        
        # Zona 2: Saprolito – nivel IC (Pgsc – s)
        # Arenisca mal gradada con grava (Altamente meteorizada)
        params_zona2 = {
            'c': 12.0,          # Cohesión (kPa)
            'phi': 32.0,        # Ángulo de fricción (grados)
            'gamma_sat': 20.0,  # Peso unitario saturado (kN/m³)
            'D_sat': 5e-6,      # Difusividad hidráulica saturada (m²/s)
            'K_sat': 1e-5,      # Conductividad hidráulica saturada (m/s)
            'theta_sat': 0.40,  # Contenido volumétrico de agua saturada
            'theta_res': 0.08,  # Contenido volumétrico de agua residual
            'alpha': 0.015      # Parámetro de van Genuchten (1/m)
        }
        
        # Retornar diccionario con parámetros por zona
        return {
            1: params_zona1,
            2: params_zona2
        }
        
    except Exception as e:
        logger.error(f"Error al cargar parámetros del suelo: {str(e)}")
        raise

def load_rainfall_data() -> Dict[str, np.ndarray]:
    """
    Carga los datos de lluvia.
    
    Returns:
        Dict[str, np.ndarray]: Diccionario con intensidades y duraciones
    """
    try:
        # Definir datos de lluvia
        intensidad = 124.60  # mm/h
        duracion = 70.0      # horas
        
        return {
            'intensities': np.array([intensidad / 1000 / 3600]),  # Convertir a m/s
            'durations': np.array([duracion * 3600])              # Convertir a segundos
        }
        
    except Exception as e:
        logger.error(f"Error al cargar datos de lluvia: {str(e)}")
        raise 