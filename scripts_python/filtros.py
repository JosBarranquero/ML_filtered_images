import cv2 as cv
import numpy as np
import math

def filtro_paso_bajo(archivo: str, salida: str, tam: int = 3):
    """Funcion que realiza un filtrado paso bajo a una imagen en escala de gris. Despues, la guarda en disco"""
    # leemos la imagen
    original = cv.imread(archivo, cv.IMREAD_UNCHANGED)

    if original is None:    # Si no ha cargado imagen, es que el archivo no existe
        raise FileNotFoundError('No se ha encontrado la imagen {0}'.format(archivo))

    if original.ndim == 3:   # Si la imagen es en color, no hacemos nada
        raise RuntimeError('¡{0} es imagen en color real!'.format(archivo))

    # Creamos la matriz del filtro paso bajo tamXtam
    # filtro = 1/tam**2 * np.ones((tam, tam))
    filtro = np.array([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])
    # filtro = np.zeros((tam, tam))
    # filtro[math.ceil(tam/2), math.ceil(tam/2)] = 1.0

    # Aplicamos el filtro
    # El segundo parametro = -1 es para mantener la misma profundidad
    resultado = cv.filter2D(original, -1, filtro)

    # Guardamos en disco
    cv.imwrite(salida, resultado)

def filtro_paso_alto(archivo: str, salida: str, tam: int = 3):
    """Funcion que realiza un filtrado paso alto a una imagen en escala de gris. Despues, la guarda en disco"""
    # leemos la imagen
    original = cv.imread(archivo, cv.IMREAD_UNCHANGED)

    if original is None:    # Si no ha cargado imagen, es que el archivo no existe
        raise FileNotFoundError('No se ha encontrado la imagen {0}'.format(archivo))

    if original.ndim == 3:   # Si la imagen es en color, no hacemos nada
        raise RuntimeError('¡{0} es imagen en color real!'.format(archivo))

    # Creamos la matriz del filtro paso alto tamXtam
    filtro = -1/tam**2 * np.ones((tam, tam))
    filtro[math.ceil(tam/2), math.ceil(tam/2)] += 1.0

    # Aplicamos el filtro
    # El segundo parametro = -1 es para mantener la misma profundidad
    resultado = cv.filter2D(original, -1, filtro)

    # Guardamos en disco
    cv.imwrite(salida, resultado)

def filtro_mediana(archivo: str, salida: str, tam: int = 3):
    """Funcion que realiza un filtrado de mediana a una imagen en escala de gris. Despues la guarda en disco"""
    # leemos la imagen
    original = cv.imread(archivo, cv.IMREAD_UNCHANGED)
    
    if original is None:    # Si no ha cargado imagen, es que el archivo no existe
        raise FileNotFoundError('No se ha encontrado la imagen {0}'.format(archivo))
    
    if original.ndim == 3:   # Si la imagen es en color, no hacemos nada
        raise RuntimeError('¡{0} es imagen en color real!'.format(archivo))

    # Aplicamos el filtro de mediana tamXtam
    resultado = cv.medianBlur(original, tam)

    # Guardamos en disco
    cv.imwrite(salida, resultado)

# Por si alguien ejecuta el script
if __name__ == '__main__':
    print("Este script no está diseñado para ser ejecutado directamente.")