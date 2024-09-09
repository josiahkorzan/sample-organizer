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
    
    if not os.path.exists(folder):
        print(f"The directory {folder} does not exist.")
        return
    
    if folder in protected_directories:
        print(f"The directory {folder} is protected and cannot be processed.")
        return
    
    get_file_type_percentages(folder)

    ans = input("Are you sure this is the folder you want to sort? (y/n)")
    if ans.lower() != 'y':
        print("Please change the destination folder")
        return
    
    # THIS IS BROKEN!!!!
    for root, dirs, files in os.walk(folder, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            # Get the name of the folder the file is currently in
            folder_name = os.path.basename(root)
            
            # Create the new file name by appending the folder name
            if folder_name not in os.path.splitext(file)[0] and folder_name != root:
                new_file_name = f"{os.path.splitext(file)[0]} : {folder_name}{os.path.splitext(file)[1]}"
                dest_path = os.path.join(folder, new_file_name)
            else:
                new_file_name = file
                dest_path = os.path.join(folder, file)
            
            # If the file already exists, append the parent folder name
            while any(os.path.exists(os.path.join(root, new_file_name)) for root, _, _ in os.walk(folder)):
                parent_folder_name = os.path.basename(os.path.dirname(root))
                new_file_name = f"{os.path.splitext(new_file_name)[0]} : {parent_folder_name}{os.path.splitext(new_file_name)[1]}"
                dest_path = os.path.join(folder, new_file_name)
                
                # Truncate the file name if it exceeds a certain length
                if len(new_file_name) > 255:
                    new_file_name = new_file_name[:255 - len(os.path.splitext(new_file_name)[1])] + os.path.splitext(new_file_name)[1]
                    dest_path = os.path.join(folder, new_file_name)
            
            shutil.move(file_path, dest_path)
            
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)

def get_file_type_percentages(folder):
    if not os.path.exists(folder):
        print(f"The directory {folder} does not exist.")
        return

    file_counts = defaultdict(int)
    total_files = 0

    for root, dirs, files in os.walk(folder):
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            file_counts[file_extension] += 1
            total_files += 1

    if total_files == 0:
        print("No files found in the directory.")
        return

    print(f"File type percentages in '{folder}'")
    for file_type, count in file_counts.items():
        percentage = (count / total_files) * 100
        print(f"{file_type}: {percentage:.2f}%")

def sort_audio_files(folder):
    # Define the allowed audio file extensions
    audio_extensions = {'.mp3', '.wav', '.aiff'}
    
    # Create the 'not_audio_files' subfolder if it doesn't exist
    not_audio_folder = os.path.join(folder, '_other')
    if not os.path.exists(not_audio_folder):
        os.makedirs(not_audio_folder)
    
    audio_folder = os.path.join(folder, '_audio_files')
    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)    
    
    
    # Traverse the directory and move non-audio files
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension not in audio_extensions:
                source_path = os.path.join(root, file)
                dest_path = os.path.join(not_audio_folder, file)
                shutil.move(source_path, dest_path)
            
            if file_extension in audio_extensions:
                source_path = os.path.join(root, file)
                dest_path = os.path.join(audio_folder, file)
                shutil.move(source_path, dest_path)
                

def main():
    parent_folder = r'/Users/josiahkorzan/Desktop/Programming/Projects/sample-organizer/test folder'
    move_files_to_parent(parent_folder)
    sort_audio_files(parent_folder)

    print("All files have been moved and subfolders deleted.")

if __name__ == "__main__":
    main()