import os
import shutil
from collections import defaultdict

# List of protected directories
protected_directories = [
    os.path.expanduser("~"),  # Home directory
    os.path.expanduser("~/Documents"),
    os.path.expanduser("~/Downloads"),
    os.path.expanduser("~/Desktop"),
    os.path.expanduser("~/Pictures"),
    os.path.expanduser("~/Music"),
    os.path.expanduser("~/Videos"),
    "/System",
    "/Library",
    "/Applications",
    "/bin",
    "/usr",
    "/sbin",
    "/etc",
    "/var",
    "/tmp"
]


def move_files_to_parent(folder):
    i = 0
    for root, dirs, files in os.walk(folder, topdown=False):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            parent_folder_name = os.path.basename(root)
            
            if parent_folder_name not in file_name:
                new_file_path = os.path.join(root, f"{file_name} : {parent_folder_name}")  # Example: prefixing 'new_' to the file name
            
            os.rename(file_path, new_file_path)
            print(f'{i}: file_path: {file_path}')
        
        # print(f'{i}: root: {root}')
        # print(f'{i}: dirs: {dirs}')
        # print(f'{i}: files: {files}')
        
        i += 1
        
def main():
    parent_folder = r'/Users/josiahkorzan/Desktop/Programming/Projects/sample-organizer/test folder'
    move_files_to_parent(parent_folder)

if __name__ == "__main__":
    main()