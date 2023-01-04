import os

def file_exists(file_name: str) -> bool:
    return os.path.isfile(file_name)

def file_count(directory: str, extension: str) -> int:
    """This function counts the files in a directory, given a certain extension"""
    num_files = 0

    # Iterating through the files
    for file in os.listdir(directory):
        # Only counting files ending in extension
        if file.endswith(extension):
            num_files += 1

    return num_files

def file_list(directory: str, extension:str='.bmp') -> list[str]:
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