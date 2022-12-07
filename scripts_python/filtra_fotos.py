import filtros as f
import utilidades as u

# Opciones para el procesado
filtra_bajo = True
filtra_alto = True
filtra_mediana = True

ruta_entrada = '../originales/'
ruta_salida  = '../filtradas/'

# Contamos las imagenes a procesar
num_imagenes = u.cuenta_archivos(ruta_entrada, '.bmp')

# Vamos procesando una a una
for i in range(1, num_imagenes + 1):
    entrada = '{0}imagen{1}.bmp'.format(ruta_entrada, i)

    if filtra_bajo:
        salida_bajo = '{0}imagen{1}-bajo.bmp'.format(ruta_salida, i)
        f.filtro_paso_bajo(entrada, salida_bajo)

    if filtra_alto:
        salida_alto = '{0}imagen{1}-alto.bmp'.format(ruta_salida, i)
        f.filtro_paso_alto(entrada, salida_alto)

    if filtra_mediana:
        salida_mediana = '{0}imagen{1}-mediana.bmp'.format(ruta_salida, i)
        f.filtro_mediana(entrada, salida_mediana)