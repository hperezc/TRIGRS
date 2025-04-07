import numpy as np
from typing import Dict, Tuple, List
from scipy import special
import logging

logger = logging.getLogger(__name__)

def calculate_pressure_head(depth: float, water_table_depth: float, 
                          soil_params: Dict, rainfall_data: Dict, 
                          time: float, flow_direction: float) -> float:
    """
    Calcula la presión de poro considerando infiltración y flujo.
    
    Args:
        depth: Profundidad del análisis (m)
        water_table_depth: Profundidad del nivel freático (m)
        soil_params: Parámetros geotécnicos del suelo
        rainfall_data: Datos de lluvia
        time: Tiempo de análisis (s)
        flow_direction: Dirección de flujo
    
    Returns:
        float: Presión de poro (m)
    """
    try:
        # Parámetros del suelo
        K_sat = soil_params['K_sat']  # Conductividad hidráulica saturada (m/s)
        D_sat = soil_params['D_sat']  # Difusividad hidráulica saturada (m²/s)
        theta_sat = soil_params['theta_sat']
        theta_res = soil_params['theta_res']
        alpha = soil_params['alpha']
        
        # Calcular infiltración acumulada
        intensities = rainfall_data['intensities']
        durations = rainfall_data['durations']
        total_duration = sum(durations)
        
        # Calcular infiltración acumulada hasta el tiempo actual
        infiltration_rate = 0
        cumulative_time = 0
        total_infiltration = 0
        
        for intensity, duration in zip(intensities, durations):
            if time > cumulative_time:
                # Calcular duración efectiva para este período
                effective_duration = min(time - cumulative_time, duration)
                current_rate = min(intensity, K_sat)
                total_infiltration += current_rate * effective_duration
                
                # Si estamos en el período actual, guardar la tasa
                if time <= cumulative_time + duration:
                    infiltration_rate = current_rate
            cumulative_time += duration
        
        # Calcular presión de poro
        if depth >= water_table_depth:
            # Por debajo del nivel freático
            pressure_head = depth - water_table_depth
        else:
            # Por encima del nivel freático
            # Presión inicial (succión)
            initial_suction = -alpha * (depth - water_table_depth)
            
            # Factor de tiempo adimensional
            T = (D_sat * time) / (depth * depth) if depth > 0 else 0
            
            # Solución de Iverson para infiltración transitoria
            beta = infiltration_rate / K_sat if K_sat > 0 else 0
            
            # Ajuste por dirección de flujo
            if not np.isnan(flow_direction):
                slope_angle = (flow_direction % 8) * 45
                slope_factor = np.cos(np.deg2rad(slope_angle))
                beta *= (0.5 + 0.5 * slope_factor)  # Mantener algo de infiltración incluso en pendientes fuertes
            
            # Durante la lluvia
            if time <= total_duration:
                # Incremento gradual de presión
                pressure_change = beta * depth * (1 - np.exp(-4 * T))  # Respuesta más gradual
                pressure_head = initial_suction + pressure_change
                
                # Efecto acumulativo del tiempo más gradual
                time_factor = (time / total_duration) ** 0.5  # Raíz cuadrada para acumulación más gradual
                pressure_head *= (1 + 1.5 * time_factor)  # Factor más moderado
            else:
                # Después de la lluvia: aumentar presión
                elapsed_time = time - total_duration
                
                # Calcular presión al final de la lluvia
                T_end = (D_sat * total_duration) / (depth * depth) if depth > 0 else 0
                end_pressure_change = beta * depth * (1 - np.exp(-4 * T_end))
                end_pressure = initial_suction + end_pressure_change
                end_pressure *= (1 + 1.5)  # Factor máximo al final de la lluvia
                
                # Aumentar presión después de la lluvia
                post_rain_factor = min(1.0 + 0.5 * (elapsed_time / (12 * 3600)), 2.0)  # Aumentar hasta 100% en 12h
                pressure_head = end_pressure * post_rain_factor
            
            # Ajuste adicional por profundidad del nivel freático
            wt_factor = np.exp(-depth / water_table_depth) if water_table_depth > 0 else 1.0
            pressure_head *= (1 + wt_factor)
        
        # Limitar presión a valores físicamente posibles
        pressure_head = np.clip(pressure_head, -10.0, depth * 2.5)  # Permitir más presión positiva
        
        return float(pressure_head)
        
    except Exception as e:
        logger.error(f"Error al calcular presión de poro: {str(e)}")
        return np.nan

def calculate_factor_of_safety(slope: float, soil_params: Dict, pressure_head: float, depth: float) -> float:
    """
    Calcula el factor de seguridad usando el método de talud infinito.
    
    Args:
        slope: Pendiente en grados
        soil_params: Parámetros geotécnicos del suelo
        pressure_head: Presión de poro (m)
        depth: Profundidad del análisis (m)
    
    Returns:
        float: Factor de seguridad
    """
    try:
        # Si la pendiente es muy cercana a cero, retornar valor máximo
        if abs(slope) < 0.001:
            return 2.0
            
        # Convertir pendiente a radianes
        slope_rad = np.deg2rad(slope)
        phi_rad = np.deg2rad(soil_params['phi'])
        
        # Parámetros del suelo
        c = soil_params['c']  # Cohesión (kPa)
        gamma_sat = soil_params['gamma_sat']  # Peso unitario saturado (kN/m³)
        gamma_w = 9.81  # Peso unitario del agua (kN/m³)
        
        # Cálculo de esfuerzos
        sigma = gamma_sat * depth * np.cos(slope_rad) * np.cos(slope_rad)  # Esfuerzo normal total
        tau = gamma_sat * depth * np.cos(slope_rad) * np.sin(slope_rad)    # Esfuerzo cortante
        
        # Presión de poros
        u = gamma_w * pressure_head if pressure_head > 0 else 0
        
        # Esfuerzo efectivo
        sigma_eff = sigma - u
        
        # Factor de seguridad (Mohr-Coulomb)
        if tau > 0:
            fs = (c + sigma_eff * np.tan(phi_rad)) / tau
        else:
            fs = 2.0
        
        # Limitar FS a un rango razonable
        fs = np.clip(fs, 0.0, 2.0)
        
        return float(fs)
        
    except Exception as e:
        logger.error(f"Error al calcular factor de seguridad: {str(e)}")
        return np.nan

def calculate_infiltration_rate(
    rainfall_intensity: float,
    K_sat: float,
    slope_angle: float
) -> float:
    """
    Calcula la tasa de infiltración efectiva.
    
    Args:
        rainfall_intensity: Intensidad de lluvia (m/s)
        K_sat: Conductividad hidráulica saturada (m/s)
        slope_angle: Ángulo de la pendiente (grados)
    
    Returns:
        float: Tasa de infiltración efectiva (m/s)
    """
    try:
        # Convertir ángulo a radianes
        alpha = np.deg2rad(slope_angle)
        
        # La infiltración efectiva es el mínimo entre la intensidad de lluvia
        # y la conductividad hidráulica saturada proyectada en la pendiente
        K_sat_effective = K_sat * np.cos(alpha)
        infiltration = min(rainfall_intensity, K_sat_effective)
        
        return max(0.0, infiltration)  # Evitar valores negativos
        
    except Exception as e:
        logger.error(f"Error en el cálculo de la tasa de infiltración: {e}")
        raise

def calculate_runoff(
    rainfall_intensity: float,
    infiltration_rate: float
) -> float:
    """
    Calcula la escorrentía superficial.
    
    Args:
        rainfall_intensity: Intensidad de lluvia (m/s)
        infiltration_rate: Tasa de infiltración (m/s)
    
    Returns:
        float: Escorrentía superficial (m/s)
    """
    try:
        return max(0.0, rainfall_intensity - infiltration_rate)
    except Exception as e:
        logger.error(f"Error en el cálculo de la escorrentía: {e}")
        raise

def analyze_grid_cell(elevation: float, slope: float, soil_params: Dict,
                     rainfall_data: Dict, initial_conditions: Dict,
                     time_steps: np.ndarray) -> Dict:
    """
    Analiza una celda del grid para todos los pasos de tiempo.
    
    Args:
        elevation: Elevación del terreno (m)
        slope: Pendiente en grados
        soil_params: Parámetros geotécnicos del suelo
        rainfall_data: Datos de lluvia
        initial_conditions: Condiciones iniciales
        time_steps: Array con los tiempos de análisis
    
    Returns:
        Dict: Resultados del análisis
    """
    try:
        # Extraer condiciones iniciales
        depth = initial_conditions['depth']
        water_table_depth = initial_conditions['water_table_depth']
        flow_direction = initial_conditions.get('flow_direction', np.nan)
        
        # Inicializar arrays de resultados
        n_steps = len(time_steps)
        results = {
            'factor_of_safety': np.full(n_steps, np.nan),
            'pressure_head': np.full(n_steps, np.nan),
            'infiltration': np.full(n_steps, np.nan),
            'runoff': np.full(n_steps, np.nan)
        }
        
        # Calcular para cada paso de tiempo
        for i, t in enumerate(time_steps):
            # Calcular presión de poro
            pressure_head = calculate_pressure_head(
                depth=depth,
                water_table_depth=water_table_depth,
                soil_params=soil_params,
                rainfall_data=rainfall_data,
                time=t,
                flow_direction=flow_direction
            )
            
            # Calcular factor de seguridad
            fs = calculate_factor_of_safety(
                slope=slope,
                soil_params=soil_params,
                pressure_head=pressure_head,
                depth=depth
            )
            
            # Calcular infiltración y escorrentía
            K_sat = soil_params['K_sat']
            current_intensity = 0
            current_time = 0
            
            for intensity, duration in zip(rainfall_data['intensities'], rainfall_data['durations']):
                if t >= current_time and t < current_time + duration:
                    current_intensity = intensity
                    break
                current_time += duration
            
            infiltration = min(current_intensity, K_sat)
            runoff = max(0, current_intensity - K_sat)
            
            # Almacenar resultados
            results['factor_of_safety'][i] = fs
            results['pressure_head'][i] = pressure_head
            results['infiltration'][i] = infiltration
            results['runoff'][i] = runoff
        
        return results
        
    except Exception as e:
        logger.error(f"Error al analizar celda: {str(e)}")
        return {key: np.full(len(time_steps), np.nan) for key in ['factor_of_safety', 'pressure_head', 'infiltration', 'runoff']}

def get_rainfall_intensity_at_time(
    rainfall_data: Dict[str, np.ndarray],
    time: float
) -> float:
    """
    Obtiene la intensidad de lluvia para un tiempo específico.
    
    Args:
        rainfall_data: Diccionario con intensidades y duraciones
        time: Tiempo para el cual se quiere la intensidad
    
    Returns:
        float: Intensidad de lluvia en el tiempo especificado
    """
    try:
        # Calcular tiempos acumulados
        cumulative_times = np.cumsum(rainfall_data['durations'])
        
        # Encontrar el período actual
        period_idx = np.searchsorted(cumulative_times, time)
        
        if period_idx >= len(rainfall_data['intensities']):
            return 0.0  # Fuera del período de lluvia
            
        return rainfall_data['intensities'][period_idx]
        
    except Exception as e:
        logger.error(f"Error al obtener intensidad de lluvia: {e}")
        raise 