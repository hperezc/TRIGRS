import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from typing import Optional, Dict
import logging
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_hillshade(dem: np.ndarray, 
                       azimuth: float = 315, 
                       angle_altitude: float = 30,  # Reducir ángulo para más contraste
                       ve: float = 2.0) -> np.ndarray:  # Factor de exageración vertical
    """
    Calcula el hillshade para un DEM.
    
    Args:
        dem: Array 2D con elevaciones
        azimuth: Ángulo azimutal de la fuente de luz (grados)
        angle_altitude: Ángulo de elevación de la fuente de luz (grados)
        ve: Factor de exageración vertical
    
    Returns:
        np.ndarray: Array 2D con valores de hillshade
    """
    # Convertir ángulos a radianes
    azimuth = np.deg2rad(azimuth)
    angle_altitude = np.deg2rad(angle_altitude)
    
    # Calcular gradientes con exageración vertical
    dx, dy = np.gradient(dem)
    dx = dx * ve  # Exagerar gradiente X
    dy = dy * ve  # Exagerar gradiente Y
    
    # Calcular pendiente y aspecto
    slope = np.pi/2. - np.arctan(np.sqrt(dx*dx + dy*dy))
    aspect = np.arctan2(-dx, dy)
    
    # Calcular hillshade
    hillshade = np.sin(angle_altitude) * np.sin(slope) + \
                np.cos(angle_altitude) * np.cos(slope) * \
                np.cos(azimuth - aspect)
    
    # Normalizar valores entre 0 y 1
    hillshade = (hillshade - hillshade.min()) / (hillshade.max() - hillshade.min())
    
    # Aumentar contraste
    hillshade = np.power(hillshade, 1.2)  # Exponente > 1 aumenta contraste
    
    return hillshade

def create_discrete_colormap() -> tuple:
    """
    Crea un mapa de colores discreto para el factor de seguridad.
    
    Returns:
        tuple: (ListedColormap, BoundaryNorm, List de patches para la leyenda)
    """
    # Definir colores y límites
    colors = ['red', 'yellow', 'green']
    boundaries = [0, 1.0, 1.5, 2.0]
    
    # Crear colormap y normalización
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N)
    
    # Crear patches para la leyenda
    legend_patches = [
        mpatches.Patch(color='red', label='FS ≤ 1.0 (Inestable)'),
        mpatches.Patch(color='yellow', label='1.0 < FS ≤ 1.5 (Marginal)'),
        mpatches.Patch(color='green', label='FS > 1.5 (Estable)')
    ]
    
    return cmap, norm, legend_patches

def generate_map(data: np.ndarray, 
                output_path: str, 
                title: str, 
                dem: Optional[np.ndarray] = None,
                result_type: str = 'factor_of_safety') -> None:
    """
    Genera un mapa con hillshade y escala continua.
    
    Args:
        data: Array con los datos a visualizar
        output_path: Ruta donde guardar el mapa
        title: Título del mapa
        dem: Array con el modelo digital de elevación
        result_type: Tipo de resultado ('factor_of_safety', 'pressure_head', 'infiltration', 'runoff')
    """
    try:
        plt.figure(figsize=(12, 8))
        
        # Generar hillshade si se proporciona DEM
        if dem is not None:
            hillshade = calculate_hillshade(dem)
            plt.imshow(hillshade, cmap='gray', alpha=0.5)
        
        # Configurar visualización según tipo de resultado
        if result_type == 'factor_of_safety':
            # Crear colormap personalizado para FS
            colors = ['darkred', 'red', 'orange', 'yellow', 'yellowgreen', 'green', 'darkgreen']
            fs_cmap = LinearSegmentedColormap.from_list('custom_fs', colors)
            
            # Plotear FS con transparencia igual que fs_time_series
            im = plt.imshow(data, cmap=fs_cmap, vmin=0.0, vmax=2.0, alpha=0.7)
            
            # Agregar colorbar con valores numéricos
            cbar = plt.colorbar(im, label='Factor de Seguridad')
            cbar.set_ticks([0.0, 0.5, 1.0, 1.5, 2.0])
            
        else:
            # Configuración para otros tipos de resultados
            if result_type == 'pressure_head':
                cmap = 'RdBu_r'
                vmin, vmax = -10, 10
                label = 'Presión de Poros (m)'
            elif result_type == 'infiltration':
                cmap = 'Blues'
                vmin, vmax = 0, np.nanmax(data)
                label = 'Infiltración (m/s)'
            else:  # runoff
                cmap = 'Blues'
                vmin, vmax = 0, np.nanmax(data)
                label = 'Escorrentía (m/s)'
            
            # Plotear datos con transparencia igual que fs_time_series
            im = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.7)
            plt.colorbar(im, label=label)
        
        # Configurar aspecto del mapa
        plt.title(title, pad=20)
        plt.xlabel('X (celdas)')
        plt.ylabel('Y (celdas)')
        
        # Ajustar layout y guardar
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Mapa guardado en: {output_path}")
        
    except Exception as e:
        logger.error(f"Error al generar mapa: {str(e)}")
        raise

def generate_time_series_plot(results: dict, 
                            output_path: str, 
                            time_steps: np.ndarray) -> None:
    """
    Genera gráficos de series temporales para los resultados.
    
    Args:
        results: Diccionario con los resultados
        output_path: Ruta donde guardar el gráfico
        time_steps: Array con los tiempos de análisis
    """
    try:
        plt.figure(figsize=(15, 10))
        
        # Convertir tiempos a horas
        times_hours = time_steps / 3600
        
        # Calcular estadísticas para cada variable
        for i, (var_name, data) in enumerate(results.items(), 1):
            plt.subplot(2, 2, i)
            
            # Calcular percentiles
            median = np.nanmedian(data, axis=(0,1))
            p25 = np.nanpercentile(data, 25, axis=(0,1))
            p75 = np.nanpercentile(data, 75, axis=(0,1))
            
            # Plotear línea media y rango
            plt.plot(times_hours, median, 'b-', label='Mediana')
            plt.fill_between(times_hours, p25, p75, color='b', alpha=0.2, label='Rango 25-75%')
            
            # Configurar gráfico
            plt.title(var_name.replace('_', ' ').title())
            plt.xlabel('Tiempo (horas)')
            plt.ylabel('Valor')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico de series temporales guardado en: {output_path}")
        
    except Exception as e:
        logger.error(f"Error al generar gráfico de series temporales: {str(e)}")
        raise

def generate_initial_conditions_plot(dem: np.ndarray,
                                  slope: np.ndarray,
                                  flow_direction: np.ndarray,
                                  depth: np.ndarray,
                                  water_table: np.ndarray,
                                  zones: np.ndarray,
                                  fs_initial: np.ndarray,
                                  output_path: str) -> None:
    """
    Genera un plot con todas las variables iniciales del análisis.
    """
    try:
        # Configurar figura
        fig = plt.figure(figsize=(12, 16))
        gs = gridspec.GridSpec(4, 2, figure=fig)
        
        # Calcular hillshade una vez
        hillshade = calculate_hillshade(dem)
        
        # 1. DEM
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(hillshade, cmap='gray', alpha=0.5)
        im1 = ax1.imshow(dem, cmap='terrain', alpha=0.7)
        plt.colorbar(im1, ax=ax1, label='Elevación [m]')
        ax1.set_title('DEM')
        ax1.set_xticks([])
        
        # 2. Pendiente
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(hillshade, cmap='gray', alpha=0.5)
        im2 = ax2.imshow(slope, cmap='RdYlGn_r', vmin=0, vmax=45, alpha=0.7)
        plt.colorbar(im2, ax=ax2, label='Pendiente [°]')
        ax2.set_title('Slope')
        ax2.set_xticks([])
        
        # 3. Dirección de flujo
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.imshow(hillshade, cmap='gray', alpha=0.5)
        im3 = ax3.imshow(flow_direction, cmap='twilight', alpha=0.7)
        plt.colorbar(im3, ax=ax3, label='Flow dir index [ESRI]')
        ax3.set_title('Flow direction')
        ax3.set_xticks([])
        
        # 4. Profundidad del suelo
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.imshow(hillshade, cmap='gray', alpha=0.5)
        # Crear máscara para valores 0
        depth_masked = np.ma.masked_where(depth == 0, depth)
        im4 = ax4.imshow(depth_masked, cmap='viridis', alpha=0.7)
        # Mostrar valores 0 en rojo
        ax4.imshow(np.where(depth == 0, 1, np.nan), cmap='Reds', alpha=0.7)
        plt.colorbar(im4, ax=ax4, label='Depth [m]')
        ax4.set_title('Zmax')
        ax4.set_xticks([])
        
        # 5. Nivel freático
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.imshow(hillshade, cmap='gray', alpha=0.5)
        # Forzar nivel freático a 50m
        water_table_plot = np.full_like(water_table, 50.0)
        water_table_plot[np.isnan(water_table)] = np.nan
        im5 = ax5.imshow(water_table_plot, cmap='Blues_r', alpha=0.7)
        plt.colorbar(im5, ax=ax5, label='Depth [m]')
        ax5.set_title('Water table')
        ax5.set_xticks([])
        
        # 6. Zonas
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.imshow(hillshade, cmap='gray', alpha=0.5)
        im6 = ax6.imshow(zones, cmap='Set2', alpha=0.7)
        plt.colorbar(im6, ax=ax6, label='Zones')
        ax6.set_title('Zones')
        ax6.set_xticks([])
        
        # 7. FS inicial
        ax7 = fig.add_subplot(gs[3, :])
        ax7.imshow(hillshade, cmap='gray', alpha=0.5)
        # Crear colormap personalizado para FS
        colors = ['darkred', 'red', 'orange', 'yellow', 'yellowgreen', 'green', 'darkgreen']
        fs_cmap = LinearSegmentedColormap.from_list('custom_fs', colors)
        im7 = ax7.imshow(fs_initial, cmap=fs_cmap, vmin=0.0, vmax=2.0, alpha=0.7)
        cbar = plt.colorbar(im7, ax=ax7, label='FS')
        cbar.set_ticks([0.0, 0.5, 1.0, 1.5, 2.0])
        ax7.set_title('FS at t=0')
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar figura
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plot de condiciones iniciales guardado en: {output_path}")
        
    except Exception as e:
        logger.error(f"Error al generar plot de condiciones iniciales: {str(e)}")
        raise

def generate_fs_time_series_plot(fs_maps: Dict[float, np.ndarray],
                               dem: Optional[np.ndarray] = None,
                               output_path: str = 'outputs/fs_time_series.png') -> None:
    """
    Genera un plot con la evolución temporal del Factor de Seguridad.
    """
    try:
        # Ordenar tiempos
        times = sorted(fs_maps.keys())
        n_times = len(times)
        
        # Calcular número de filas y columnas
        n_cols = 4  # Aumentamos a 4 columnas para mejor distribución
        n_rows = (n_times + n_cols - 1) // n_cols
        
        # Configurar figura con tamaño ajustado
        fig = plt.figure(figsize=(20, 5*n_rows))  # Aumentamos el ancho
        
        # Generar hillshade si se proporciona DEM
        hillshade = None
        if dem is not None:
            hillshade = calculate_hillshade(dem)
        
        # Crear colormap personalizado para FS
        colors = ['darkred', 'red', 'orange', 'yellow', 'yellowgreen', 'green', 'darkgreen']
        fs_cmap = LinearSegmentedColormap.from_list('custom_fs', colors)
        
        # Plotear cada mapa de FS
        for i, time in enumerate(times):
            ax = fig.add_subplot(n_rows, n_cols, i+1)
            
            # Agregar hillshade si existe
            if hillshade is not None:
                ax.imshow(hillshade, cmap='gray', alpha=0.5)
            
            # Plotear FS
            im = ax.imshow(fs_maps[time], cmap=fs_cmap, vmin=0.0, vmax=2.0, alpha=0.7)
            
            # Configurar título y ejes
            ax.set_title(f'FS at t={time:.1f} h')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Agregar colorbar
            cbar = plt.colorbar(im, ax=ax, label='FS')
            cbar.set_ticks([0.0, 0.5, 1.0, 1.5, 2.0])
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar figura
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plot de serie temporal de FS guardado en: {output_path}")
        
    except Exception as e:
        logger.error(f"Error al generar plot de serie temporal de FS: {str(e)}")
        raise

def generate_statistical_analysis(results: Dict[str, np.ndarray], 
                                time_steps: np.ndarray,
                                slope: np.ndarray,
                                zones: np.ndarray,
                                output_dir: str) -> None:
    """
    Genera análisis estadístico completo de los resultados del modelo TRIGRS.
    """
    try:
        # Convertir tiempos a horas
        times_hours = time_steps / 3600
        
        # 1. Series Temporales con Rangos
        # Usar estilo predeterminado más limpio de matplotlib
        plt.style.use('default')  # Cambiamos seaborn por default
        
        # Configurar el estilo general de las gráficas
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['axes.titlesize'] = 12
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Factor de Seguridad
        fs_median = np.nanmedian(results['factor_of_safety'], axis=(0,1))
        fs_25 = np.nanpercentile(results['factor_of_safety'], 25, axis=(0,1))
        fs_75 = np.nanpercentile(results['factor_of_safety'], 75, axis=(0,1))
        
        ax1.plot(times_hours, fs_median, 'b-', linewidth=2, label='Mediana')
        ax1.fill_between(times_hours, fs_25, fs_75, alpha=0.3, color='blue', label='Rango 25-75%')
        ax1.set_title('Factor de Seguridad')
        ax1.set_xlabel('Tiempo (horas)')
        ax1.set_ylabel('FS')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Presión de Poros
        ph_median = np.nanmedian(results['pressure_head'], axis=(0,1))
        ph_25 = np.nanpercentile(results['pressure_head'], 25, axis=(0,1))
        ph_75 = np.nanpercentile(results['pressure_head'], 75, axis=(0,1))
        
        ax2.plot(times_hours, ph_median, 'b-', linewidth=2, label='Mediana')
        ax2.fill_between(times_hours, ph_25, ph_75, alpha=0.3, color='blue', label='Rango 25-75%')
        ax2.set_title('Presión de Poros')
        ax2.set_xlabel('Tiempo (horas)')
        ax2.set_ylabel('Presión (m)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Infiltración
        inf_median = np.nanmedian(results['infiltration'], axis=(0,1))
        inf_25 = np.nanpercentile(results['infiltration'], 25, axis=(0,1))
        inf_75 = np.nanpercentile(results['infiltration'], 75, axis=(0,1))
        
        ax3.plot(times_hours, inf_median, 'b-', linewidth=2, label='Mediana')
        ax3.fill_between(times_hours, inf_25, inf_75, alpha=0.3, color='blue', label='Rango 25-75%')
        ax3.set_title('Infiltración')
        ax3.set_xlabel('Tiempo (horas)')
        ax3.set_ylabel('Tasa (m/s)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Área Inestable
        unstable_area = np.sum(results['factor_of_safety'] < 1.0, axis=(0,1)) / np.sum(~np.isnan(results['factor_of_safety'][:,:,0])) * 100
        
        ax4.plot(times_hours, unstable_area, 'r-', linewidth=2, label='Área Inestable')
        ax4.set_title('Evolución del Área Inestable')
        ax4.set_xlabel('Tiempo (horas)')
        ax4.set_ylabel('Porcentaje (%)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/time_series_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Análisis de Correlación
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # FS vs Presión de Poros - Ahora usando scatter
        ax1.scatter(results['pressure_head'].flatten(), 
                   results['factor_of_safety'].flatten(),
                   alpha=0.1,  # Transparencia para ver densidad
                   s=1,       # Tamaño pequeño de puntos
                   c='blue')  # Color azul
        ax1.set_xlabel('Presión de Poros (m)')
        ax1.set_ylabel('Factor de Seguridad')
        ax1.set_title('FS vs Presión de Poros')
        ax1.grid(True, alpha=0.3)
        # Agregar línea de FS = 1
        ax1.axhline(y=1.0, color='r', linestyle='--', label='FS = 1.0')
        ax1.legend()
        
        # FS vs Pendiente - Ahora usando scatter
        ax2.scatter(np.repeat(slope[:,:,np.newaxis], len(time_steps), axis=2).flatten(),
                   results['factor_of_safety'].flatten(),
                   alpha=0.1,  # Transparencia para ver densidad
                   s=1,       # Tamaño pequeño de puntos
                   c='blue')  # Color azul
        ax2.set_xlabel('Pendiente (°)')
        ax2.set_ylabel('Factor de Seguridad')
        ax2.set_title('FS vs Pendiente')
        ax2.grid(True, alpha=0.3)
        # Agregar línea de FS = 1
        ax2.axhline(y=1.0, color='r', linestyle='--', label='FS = 1.0')
        ax2.legend()
        
        # Histograma de FS
        ax3.hist(results['factor_of_safety'].flatten(), bins=50, 
                density=True, alpha=0.7, color='blue', edgecolor='black')
        ax3.axvline(x=1.0, color='r', linestyle='--', linewidth=2, label='FS = 1.0')
        ax3.set_xlabel('Factor de Seguridad')
        ax3.set_ylabel('Densidad')
        ax3.set_title('Distribución del FS')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Box plot por zona
        zone_data = []
        zone_labels = []
        for zone_id in np.unique(zones[~np.isnan(zones)]):
            zone_mask = zones == zone_id
            zone_fs = results['factor_of_safety'][zone_mask]
            zone_data.append(zone_fs.flatten())
            zone_labels.append(f'Zona {int(zone_id)}')
        
        ax4.boxplot(zone_data, labels=zone_labels)
        ax4.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='FS = 1.0')
        ax4.set_ylabel('Factor de Seguridad')
        ax4.set_title('FS por Zona')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Estadísticas por zona
        stats_file = f'{output_dir}/zone_statistics.txt'
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("Estadísticas por Zona\n")
            f.write("====================\n\n")
            
            for zone_id in np.unique(zones[~np.isnan(zones)]):
                zone_mask = zones == zone_id
                zone_fs = results['factor_of_safety'][zone_mask]
                zone_ph = results['pressure_head'][zone_mask]
                
                f.write(f"\nZona {int(zone_id)}:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Factor de Seguridad:\n")
                f.write(f"  Media: {np.nanmean(zone_fs):.3f}\n")
                f.write(f"  Mediana: {np.nanmedian(zone_fs):.3f}\n")
                f.write(f"  Desv. Est.: {np.nanstd(zone_fs):.3f}\n")
                f.write(f"  Mínimo: {np.nanmin(zone_fs):.3f}\n")
                f.write(f"  Máximo: {np.nanmax(zone_fs):.3f}\n")
                f.write(f"  % Área Inestable: {np.sum(zone_fs < 1.0) / np.sum(~np.isnan(zone_fs)) * 100:.1f}%\n")
                
                f.write(f"\nPresión de Poros:\n")
                f.write(f"  Media: {np.nanmean(zone_ph):.3f} m\n")
                f.write(f"  Mediana: {np.nanmedian(zone_ph):.3f} m\n")
                f.write(f"  Desv. Est.: {np.nanstd(zone_ph):.3f} m\n")
                f.write(f"  Mínimo: {np.nanmin(zone_ph):.3f} m\n")
                f.write(f"  Máximo: {np.nanmax(zone_ph):.3f} m\n\n")
        
        logger.info(f"Análisis estadístico completado y guardado en: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error al generar análisis estadístico: {str(e)}")
        raise

def generate_model_validation_plots(results: Dict[str, np.ndarray], 
                                  time_steps: np.ndarray,
                                  slope: np.ndarray,
                                  zones: np.ndarray,
                                  dem: np.ndarray,
                                  output_dir: str) -> None:
    """
    Genera gráficos de validación con mejoras en visualización.
    """
    try:
        times_hours = time_steps / 3600
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Presión de Poros vs Infiltración Acumulada con intervalos de confianza
        ph_mean = np.nanmean(results['pressure_head'], axis=(0,1))
        ph_std = np.nanstd(results['pressure_head'], axis=(0,1))
        inf_acum = np.cumsum(results['infiltration'], axis=2)
        inf_acum_mean = np.nanmean(inf_acum, axis=(0,1))
        inf_acum_std = np.nanstd(inf_acum, axis=(0,1))
        
        ax1.plot(times_hours, ph_mean, 'b-', label='Presión de Poros')
        ax1.fill_between(times_hours, 
                        ph_mean - ph_std, 
                        ph_mean + ph_std, 
                        color='b', alpha=0.2, label='±1 Desv. Est.')
        ax1.set_xlabel('Tiempo (horas)')
        ax1.set_ylabel('Presión de Poros Media (m)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(times_hours, inf_acum_mean, 'r-', label='Infiltración Acumulada')
        ax1_twin.fill_between(times_hours, 
                            inf_acum_mean - inf_acum_std,
                            inf_acum_mean + inf_acum_std,
                            color='r', alpha=0.2)
        ax1_twin.set_ylabel('Infiltración Acumulada (m)', color='r')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        ax1.set_title('Presión de Poros vs Infiltración Acumulada')
        
        # Combinar leyendas
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 2. FS vs Elevación con intervalos de confianza
        elevation_bins = np.linspace(np.nanmin(dem), np.nanmax(dem), 20)
        fs_means = []
        fs_stds = []
        
        # Seleccionar tiempos específicos para mejor visualización
        selected_times = [0, 5, 24, 60]  # Tiempos clave en horas
        selected_indices = [np.where(time_steps/3600 == t)[0][0] for t in selected_times]
        
        for t in selected_indices:
            means = []
            stds = []
            for i in range(len(elevation_bins)-1):
                mask = (dem >= elevation_bins[i]) & (dem < elevation_bins[i+1])
                fs_values = results['factor_of_safety'][:,:,t][mask]
                means.append(np.nanmean(fs_values))
                stds.append(np.nanstd(fs_values))
            
            bin_centers = (elevation_bins[:-1] + elevation_bins[1:]) / 2
            ax2.plot(bin_centers, means, 
                    label=f't={time_steps[t]/3600:.0f}h',
                    marker='o')
            ax2.fill_between(bin_centers,
                           np.array(means) - np.array(stds),
                           np.array(means) + np.array(stds),
                           alpha=0.2)
        
        ax2.set_xlabel('Elevación (m)')
        ax2.set_ylabel('Factor de Seguridad')
        ax2.set_title('FS vs Elevación')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. FS Promedio vs Pendiente con intervalos de confianza
        slope_bins = np.linspace(0, 90, 10)
        slope_centers = (slope_bins[:-1] + slope_bins[1:]) / 2
        fs_means = []
        fs_stds = []
        
        for i in range(len(slope_bins)-1):
            mask = (slope >= slope_bins[i]) & (slope < slope_bins[i+1])
            fs_values = results['factor_of_safety'][mask]
            fs_means.append(np.nanmean(fs_values))
            fs_stds.append(np.nanstd(fs_values))
        
        ax3.plot(slope_centers, fs_means, 'b-o')
        ax3.fill_between(slope_centers,
                        np.array(fs_means) - np.array(fs_stds),
                        np.array(fs_means) + np.array(fs_stds),
                        color='b', alpha=0.2, label='±1 Desv. Est.')
        ax3.set_xlabel('Pendiente (°)')
        ax3.set_ylabel('FS Promedio')
        ax3.set_title('FS Promedio vs Pendiente')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Balance Hídrico (solo volúmenes acumulados)
        inf_total = np.nansum(results['infiltration'], axis=(0,1))
        runoff_total = np.nansum(results['runoff'], axis=(0,1))

        # Calcular volúmenes acumulados
        inf_acum = np.cumsum(inf_total)
        runoff_acum = np.cumsum(runoff_total)

        # Plotear volúmenes acumulados
        ax4.plot(times_hours, inf_acum, 'b-', 
                label='Infiltración Acumulada', linewidth=2)
        ax4.plot(times_hours, runoff_acum, 'r-', 
                label='Escorrentía Acumulada', linewidth=2)
        ax4.set_xlabel('Tiempo (horas)')
        ax4.set_ylabel('Volumen Acumulado (m³)')
        ax4.set_title('Balance Hídrico Acumulado')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_validation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error al generar gráficos de validación: {str(e)}")
        raise