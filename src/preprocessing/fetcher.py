import os

def list_dir_files(filepath: str) -> list[str]:
    names = []
    for dirpath, _, filenames in os.walk(filepath):
        for filename in filenames:
            names.append(os.path.join(dirpath, filename))
    return names
