import os

def existe_archvio(archivo: str) -> bool:
    return os.path.isfile(archivo)

def cuenta_archivos(directorio: str, extension: str) -> int:
    """Funcion que cuenta los archivos de un directorio con una cierta extension"""
    num_archivos = 0

    # Recorremos los archivos que hay en el directorio
    for archivo in os.listdir(directorio):
        # Solamente nos interesan los archivos con nuestra extension
        if archivo.endswith(extension):
            num_archivos += 1

    return num_archivos

def lista_archivos(directorio: str, extension:str='.bmp') -> list[str]:
    """Funcion que busca los archivos de un directorio con una extension y devuelve una lista con ellos"""
    archivos = list()

        # Recorremos los archivos que hay en el directorio
    for archivo in os.listdir(directorio):
        # Solamente nos interesan los archivos con nuestra extension
        if os.path.isfile(directorio + archivo) and archivo.endswith(extension):
            archivos.append(archivo)

    return archivos
    
# Por si alguien ejecuta el script
if __name__ == '__main__':
    print("Este script no está diseñado para ser ejecutado directamente.")