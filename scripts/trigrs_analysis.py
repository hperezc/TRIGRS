import numpy as np
from typing import Dict, Any
import logging
from trigrs_calculations import (
    calculate_pressure_head,
    calculate_factor_of_safety,
    calculate_infiltration_rate,
    calculate_runoff
)

logger = logging.getLogger(__name__)

def run_analysis(
    dem: np.ndarray,
    slope: np.ndarray,
    zones: np.ndarray,
    flow_direction: np.ndarray,
    depth: np.ndarray,
    water_table: np.ndarray,
    soil_params: Dict[int, Dict[str, float]],
    rainfall_data: Dict[str, np.ndarray],
    time_steps: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Ejecuta el análisis TRIGRS.
    
    Args:
        dem: Modelo digital de elevación
        slope: Pendientes
        zones: Zonas de suelo
        flow_direction: Dirección de flujo
        depth: Profundidad del suelo
        water_table: Nivel freático
        soil_params: Parámetros geotécnicos por zona
        rainfall_data: Datos de lluvia
        time_steps: Tiempos de análisis
    
    Returns:
        Dict[str, np.ndarray]: Resultados del análisis
    """
    try:
        # Obtener dimensiones
        nrows, ncols = dem.shape
        n_steps = len(time_steps)
        
        # Inicializar arrays de resultados
        results = {
            'factor_of_safety': np.full((nrows, ncols, n_steps), np.nan),
            'pressure_head': np.full((nrows, ncols, n_steps), np.nan),
            'infiltration': np.full((nrows, ncols, n_steps), np.nan),
            'runoff': np.full((nrows, ncols, n_steps), np.nan)
        }
        
        # Calcular zonas de convergencia
        dx, dy = np.gradient(dem)
        slope_mag = np.sqrt(dx**2 + dy**2)
        convergence = np.zeros_like(dem)
        
        # Calcular convergencia basada en cambios de pendiente
        for i in range(1, nrows-1):
            for j in range(1, ncols-1):
                # Diferencia de pendiente con celdas vecinas
                diff = slope_mag[i,j] - np.mean([
                    slope_mag[i-1,j], slope_mag[i+1,j],
                    slope_mag[i,j-1], slope_mag[i,j+1]
                ])
                convergence[i,j] = max(0, diff)
        
        # Normalizar convergencia
        convergence = convergence / np.nanmax(convergence)
        
        # Contadores para diagnóstico
        total_cells = nrows * ncols
        processed_cells = 0
        skipped_cells = 0
        nan_counts = {
            'dem': 0,
            'slope': 0,
            'zones': 0,
            'flow_direction': 0,
            'depth': 0,
            'water_table': 0
        }
        
        # Configurar tiempos de análisis
        #time_steps = np.array([0, 1, 2, 3, 4, 5, 6, 12, 24, 48, 54, 60]) * 3600  # Convertir a segundos
        
        # Analizar cada celda
        for i in range(nrows):
            for j in range(ncols):
                # Verificar valores NaN y profundidad cero
                if np.isnan(dem[i, j]):
                    nan_counts['dem'] += 1
                    continue
                if np.isnan(slope[i, j]):
                    nan_counts['slope'] += 1
                    continue
                if np.isnan(zones[i, j]):
                    nan_counts['zones'] += 1
                    continue
                if np.isnan(flow_direction[i, j]):
                    nan_counts['flow_direction'] += 1
                    continue
                if np.isnan(depth[i, j]):
                    nan_counts['depth'] += 1
                    continue
                if np.isnan(water_table[i, j]):
                    nan_counts['water_table'] += 1
                    continue
                    
                # Si la profundidad es 0, marcar como estable y continuar
                if depth[i, j] == 0:
                    # Marcar como estable (FS = 2.0) y sin infiltración
                    for k in range(n_steps):
                        results['factor_of_safety'][i, j, k] = 2.0
                        results['pressure_head'][i, j, k] = 0.0
                        results['infiltration'][i, j, k] = 0.0
                        results['runoff'][i, j, k] = rainfall_data['intensities'][0]  # Toda la lluvia es escorrentía
                    processed_cells += 1
                    continue
                
                # Obtener parámetros del suelo para la zona
                zone = int(zones[i, j])
                if zone not in soil_params:
                    skipped_cells += 1
                    continue
                
                # Calcular factores de ajuste basados en condiciones locales
                slope_factor = np.clip(slope[i,j] / 45.0, 0, 1)  # Normalizar pendiente
                depth_factor = np.clip(depth[i,j] / 10.0, 0, 1)  # Normalizar profundidad
                conv_factor = convergence[i,j]  # Factor de convergencia
                
                # Factor combinado para ajustar la presión de poros
                pressure_adjustment = 1.0 - 0.5 * (slope_factor + depth_factor + conv_factor) / 3
                
                # Calcular para cada paso de tiempo
                for k, time in enumerate(time_steps):
                    # Calcular presión de poros base
                    pressure_head = calculate_pressure_head(
                        depth=depth[i, j],
                        water_table_depth=water_table[i, j],
                        soil_params=soil_params[zone],
                        rainfall_data=rainfall_data,
                        time=time,
                        flow_direction=flow_direction[i, j]
                    )
                    
                    # Ajustar presión de poros según condiciones locales
                    pressure_head *= pressure_adjustment
                    
                    # Limitar presión máxima según profundidad
                    max_pressure = depth[i,j] * 0.8  # Máximo 80% de la profundidad
                    pressure_head = min(pressure_head, max_pressure)
                    
                    # Calcular factor de seguridad
                    fs = calculate_factor_of_safety(
                        slope=slope[i, j],
                        soil_params=soil_params[zone],
                        pressure_head=pressure_head,
                        depth=depth[i, j]
                    )
                    
                    # Calcular infiltración y escorrentía
                    K_sat = soil_params[zone]['K_sat']
                    current_intensity = 0
                    current_time = 0
                    
                    for intensity, duration in zip(rainfall_data['intensities'], rainfall_data['durations']):
                        if time >= current_time and time < current_time + duration:
                            current_intensity = intensity
                            break
                        current_time += duration
                    
                    # Ajustar conductividad según convergencia
                    K_sat *= (1.0 - 0.3 * conv_factor)  # Reducir hasta 30% en zonas convergentes
                    
                    infiltration = calculate_infiltration_rate(
                        rainfall_intensity=current_intensity,
                        K_sat=K_sat,
                        slope_angle=slope[i, j]
                    )
                    
                    runoff = calculate_runoff(
                        rainfall_intensity=current_intensity,
                        infiltration_rate=infiltration
                    )
                    
                    # Almacenar resultados
                    results['factor_of_safety'][i, j, k] = fs
                    results['pressure_head'][i, j, k] = pressure_head
                    results['infiltration'][i, j, k] = infiltration
                    results['runoff'][i, j, k] = runoff
                
                processed_cells += 1
        
        # Imprimir resumen
        print("\nResumen del análisis:")
        print(f"Total de celdas: {total_cells}")
        print(f"Celdas procesadas: {processed_cells}")
        print(f"Celdas omitidas: {skipped_cells}")
        print(f"Celdas con DEM NaN: {nan_counts['dem']}")
        print(f"Celdas con pendiente NaN: {nan_counts['slope']}")
        print(f"Celdas con zona NaN: {nan_counts['zones']}")
        print(f"Celdas con dirección NaN: {nan_counts['flow_direction']}")
        print(f"Celdas con profundidad NaN: {nan_counts['depth']}")
        print(f"Celdas con nivel freático NaN: {nan_counts['water_table']}")
        print("Análisis completado exitosamente\n")
        
        return results
        
    except Exception as e:
        logger.error(f"Error en el análisis TRIGRS: {str(e)}")
        raise 