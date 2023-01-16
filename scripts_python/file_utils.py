import os
import random

def randomIndexList(a_list: list[str]) -> list[int]:
    """This funcion returns a list of indices in random order"""
    max_value = len(a_list)
    
    # Getting a list of max_value random unrepeated numbers between 0 and max_value
    indices = random.sample(range(0, max_value), max_value)

    return indices
        
def fileExists(file_name: str) -> bool:
    """This function checks if a given file exists"""
    return os.path.isfile(file_name)

def fileCount(directory: str, extension: str) -> int:
    """This function counts the files in a directory, given a certain extension"""
    num_files = 0

    # Iterating through the files
    for file in os.listdir(directory):
        # Only counting files ending in extension
        if file.endswith(extension):
            num_files += 1

    return num_files

def fileList(directory: str, extension:str='.bmp') -> list[str]:
    """This function looks for files in a directory, given a certain extension. Then, it returns them in a list"""
    files = list()

    # Iterating through the files
    for file in os.listdir(directory):
        # Only keeping track of files ending in extension
        if os.path.isfile(directory + file) and file.endswith(extension):
            files.append(file)

    return files
    
# Show a message if the script is run by itself
if __name__ == '__main__':
    print("This script is not desingned to be standalone.")