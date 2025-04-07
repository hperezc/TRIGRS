import numpy as np

def leer_asc(archivo):
    """Lee un archivo ASC y retorna el encabezado y los datos"""
    with open(archivo, 'r') as f:
        # Leer encabezado
        ncols = int(f.readline().split()[1])
        nrows = int(f.readline().split()[1])
        xllcorner = float(f.readline().split()[1])
        yllcorner = float(f.readline().split()[1])
        cellsize = float(f.readline().split()[1])
        nodata = float(f.readline().split()[1])
        
        # Leer datos como una lista de líneas
        lineas = f.readlines()
        datos = []
        for linea in lineas:
            valores = [float(x) for x in linea.strip().split()]
            if valores:  # Ignorar líneas vacías
                datos.append(valores)
        datos = np.array(datos)
        
    header = {
        'ncols': ncols,
        'nrows': nrows,
        'xllcorner': xllcorner,
        'yllcorner': yllcorner,
        'cellsize': cellsize,
        'NODATA_value': -9999  # Cambiar 'nodata' a 'NODATA_value' para coincidir con el formato estándar
    }
    
    # Reemplazar el valor NODATA original con -9999
    datos[datos == nodata] = -9999
    # Asegurar que los valores NaN también se conviertan a -9999
    datos = np.nan_to_num(datos, nan=-9999)
    
    return header, datos

def guardar_asc(archivo, header, datos):
    """Guarda los datos en formato ASC"""
    with open(archivo, 'w') as f:
        f.write(f"ncols         {header['ncols']}\n")
        f.write(f"nrows         {header['nrows']}\n")
        f.write(f"xllcorner     {header['xllcorner']}\n")
        f.write(f"yllcorner     {header['yllcorner']}\n")
        f.write(f"cellsize      {header['cellsize']}\n")
        f.write(f"NODATA_value  {header['NODATA_value']}\n")  # Actualizar para usar NODATA_value
        np.savetxt(f, datos, fmt='%.2f', delimiter=' ')

def verificar_dimensiones(datos, header):
    """Verifica y corrige las dimensiones de los datos"""
    if datos.shape != (header['nrows'], header['ncols']):
        print(f"Advertencia: Las dimensiones no coinciden. Esperado: {(header['nrows'], header['ncols'])}, Actual: {datos.shape}")
        # Ajustar las dimensiones si es necesario
        if len(datos.shape) == 1:
            datos = datos.reshape(header['nrows'], header['ncols'])
        else:
            # Recortar o rellenar según sea necesario
            new_data = np.full((header['nrows'], header['ncols']), header['NODATA_value'])
            rows = min(datos.shape[0], header['nrows'])
            cols = min(datos.shape[1], header['ncols'])
            new_data[:rows, :cols] = datos[:rows, :cols]
            datos = new_data
    return datos

def preprocesar_archivos():
    try:
        # Leer archivos originales
        print("Leyendo archivos originales...")
        dem_header, dem_data = leer_asc('TRIGRS/inputs/dem.asc')
        slope_header, slope_data = leer_asc('TRIGRS/inputs/slope.asc')
        zones_header, zones_data = leer_asc('TRIGRS/inputs/zones.asc')
        
        # Verificar y mostrar dimensiones
        print(f"Dimensiones DEM: {dem_data.shape}")
        print(f"Dimensiones Slope: {slope_data.shape}")
        print(f"Dimensiones Zones: {zones_data.shape}")
        
        # Usar las dimensiones del DEM como referencia
        ref_header = dem_header.copy()
        print(f"Usando dimensiones de referencia: {ref_header['nrows']}x{ref_header['ncols']}")
        
        # Verificar y ajustar dimensiones
        dem_data = verificar_dimensiones(dem_data, ref_header)
        slope_data = verificar_dimensiones(slope_data, ref_header)
        zones_data = verificar_dimensiones(zones_data, ref_header)
        
        # Asegurar que los valores estén en rangos válidos
        dem_data = np.where(dem_data == -9999, -9999, np.clip(dem_data, 0, 99999))
        slope_data = np.where(slope_data == -9999, -9999, np.clip(slope_data, 0, 90))
        zones_data = np.where(zones_data == -9999, -9999, np.clip(zones_data, 1, 99999))
        
        # Verificar dimensiones finales
        print("\nDimensiones finales después del procesamiento:")
        print(f"DEM: {dem_data.shape}")
        print(f"Slope: {slope_data.shape}")
        print(f"Zones: {zones_data.shape}")
        
        # Guardar archivos procesados
        print("\nGuardando archivos procesados...")
        guardar_asc('TRIGRS/inputs/dem_proc.asc', ref_header, dem_data)
        guardar_asc('TRIGRS/inputs/slope_proc.asc', ref_header, slope_data)
        guardar_asc('TRIGRS/inputs/zones_proc.asc', ref_header, zones_data)
        print("Procesamiento completado exitosamente.")
        
    except Exception as e:
        print(f"Error durante el preprocesamiento: {str(e)}")

if __name__ == '__main__':
    preprocesar_archivos() 