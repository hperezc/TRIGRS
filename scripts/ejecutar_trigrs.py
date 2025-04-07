from trigrs_model import TRIGRSModel
import os
from pathlib import Path
import numpy as np
import visualization
import logging
import trigrs_io
import trigrs_analysis

# Configurar logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Crear handler para consola si no existe
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def ejecutar_analisis_trigrs():
    try:
        # Crear directorio outputs si no existe
        output_dir = Path('outputs')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cargar datos de entrada
        input_data = trigrs_io.load_input_data()
        dem = input_data['dem']
        slope = input_data['slope']
        zones = input_data['zones']
        flow_direction = input_data['flow_direction']
        depth = input_data['depth']
        water_table = input_data['water_table']
        
        # Configurar parámetros geotécnicos
        soil_params = trigrs_io.load_soil_params()
        
        # Configurar datos de lluvia
        rainfall_data = trigrs_io.load_rainfall_data()
        
        # Configurar tiempos de análisis
        time_steps = np.array([0, 1, 2, 3, 4, 5, 6, 12, 24, 48, 54, 60]) * 3600  # Convertir a segundos
        
        # Ejecutar análisis TRIGRS
        results = trigrs_analysis.run_analysis(
            dem=dem,
            slope=slope,
            zones=zones,
            flow_direction=flow_direction,
            depth=depth,
            water_table=water_table,
            soil_params=soil_params,
            rainfall_data=rainfall_data,
            time_steps=time_steps
        )
        
        # Generar visualizaciones
        logger.info("Generando visualizaciones...")
        
        # 1. Plot de condiciones iniciales
        visualization.generate_initial_conditions_plot(
            dem=dem,
            slope=slope,
            flow_direction=flow_direction,
            depth=depth,
            water_table=water_table,
            zones=zones,
            fs_initial=results['factor_of_safety'][:, :, 0],  # t=0
            output_path=str(output_dir / 'initial_conditions.png')
        )
        
        # 2. Plot de evolución temporal del FS
        fs_maps = {
            time/3600: results['factor_of_safety'][:, :, i]  # Convertir tiempo a horas
            for i, time in enumerate(time_steps)
        }
        
        visualization.generate_fs_time_series_plot(
            fs_maps=fs_maps,
            dem=dem,
            output_path=str(output_dir / 'fs_time_series.png')
        )
        
        # 3. Generar mapas individuales (ya existente)
        for i, time in enumerate(time_steps):
            # Factor de Seguridad
            visualization.generate_map(
                data=results['factor_of_safety'][:, :, i],
                output_path=str(output_dir / f'fs_t{time/3600:.1f}h.png'),
                title=f'Factor de Seguridad (t={time/3600:.1f}h)',
                dem=dem,
                result_type='factor_of_safety'
            )
            
            # Presión de poros
            visualization.generate_map(
                data=results['pressure_head'][:, :, i],
                output_path=str(output_dir / f'ph_t{time/3600:.1f}h.png'),
                title=f'Presión de Poros (t={time/3600:.1f}h)',
                dem=dem,
                result_type='pressure_head'
            )
            
            # Infiltración
            visualization.generate_map(
                data=results['infiltration'][:, :, i],
                output_path=str(output_dir / f'inf_t{time/3600:.1f}h.png'),
                title=f'Infiltración (t={time/3600:.1f}h)',
                dem=dem,
                result_type='infiltration'
            )
            
            # Escorrentía
            visualization.generate_map(
                data=results['runoff'][:, :, i],
                output_path=str(output_dir / f'runoff_t{time/3600:.1f}h.png'),
                title=f'Escorrentía (t={time/3600:.1f}h)',
                dem=dem,
                result_type='runoff'
            )
        
        # 4. Generar análisis estadístico
        visualization.generate_statistical_analysis(
            results=results,
            time_steps=time_steps,
            slope=slope,
            zones=zones,
            output_dir=str(output_dir)
        )
        
        # Después de generar los mapas individuales
        selected_times = [0, 3, 6, 12, 24, 60]  # Tiempos clave en horas
        selected_indices = [np.where(time_steps/3600 == t)[0][0] for t in selected_times]
        visualization.generate_model_validation_plots(
            results=results,
            time_steps=time_steps,
            slope=slope,
            zones=zones,
            dem=dem,
            output_dir=str(output_dir)
        )
        
        logger.info("Análisis TRIGRS completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error en el análisis TRIGRS: {str(e)}")
        raise

if __name__ == "__main__":
    ejecutar_analisis_trigrs() 