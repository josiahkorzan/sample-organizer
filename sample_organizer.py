import os
import shutil
from collections import defaultdict

import time

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
    
    for root, dirs, files in os.walk(folder, topdown=False):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            parent_folder_name = os.path.basename(root)
            
            # Split the file name and extension
            name, ext = os.path.splitext(file_name)
            
            # Create the new file name by appending the folder name
            if parent_folder_name not in file_name:
                new_file_path = os.path.join(root, f"{name}__{parent_folder_name}{ext}") 
                os.rename(file_path, new_file_path)
    
    for root, dirs, files in os.walk(folder, topdown=False):    
        for file_name in files:
            file_path = os.path.join(root, file_name)
            dest_path = os.path.join(folder, file_name)    
            shutil.move(file_path, dest_path)
            print(f"file: {file_path}, moved")
            
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)

def copy_files_to_output(parent_folder, output_folder):      
    if not os.path.exists(parent_folder):
            print(f"The directory {parent_folder} does not exist.")
            return
        
    if parent_folder in protected_directories:
        print(f"The directory {parent_folder} is protected and cannot be processed.")
        return
    
    get_file_type_percentages(parent_folder)

    ans = input("Are you sure this is the folder you want to sort? (y/n)")
    if ans.lower() != 'y':
        print("Please change the destination folder")
        return
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for root, dirs, files in os.walk(parent_folder, topdown=False):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            parent_folder_name = os.path.basename(root)
            
            # Split the file name and extension
            name, ext = os.path.splitext(file_name)
            
            # Create the new file name by appending the folder name
            if parent_folder_name not in file_name and ' __ ' not in file_name:
                new_file_name = f"{name} __ {parent_folder_name}{ext}"
            else:
                new_file_name = file_name
            
            dest_path = os.path.join(output_folder, new_file_name)
            shutil.copyfile(file_path, dest_path)
            
            print(f"file: {file_path}, copied as {dest_path}")

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

def sort_audio_files_by_file_type(folder):
    # Define the allowed audio file extensions
    audio_extensions = {'.wav', '.mp3', '.mp4', '.flac', '.aiff'}
    
    # Create the 'not_audio_files' subfolder if it doesn't exist
    not_audio_folder = os.path.join(folder, 'other')
    if not os.path.exists(not_audio_folder):
        os.makedirs(not_audio_folder)
    
    audio_folder = os.path.join(folder, 'wav_files')
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
                print(f"moved {source_path} to wav files")
            
            if file_extension in audio_extensions:
                source_path = os.path.join(root, file)
                dest_path = os.path.join(audio_folder, file)
                shutil.move(source_path, dest_path)
                print(f"moved {source_path} to other")
   
def sort_audio_files_by_name(output_folder, audio_folder, bins: dict):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    audio_folder = f"{audio_folder}/wav_files"
    
    # Ensure the audio_folder exists
    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)
    
    for audio_file in os.listdir(audio_folder):
        file_path = os.path.join(audio_folder, audio_file)
        if os.path.isfile(file_path):
            file_name = os.path.basename(file_path).lower()
            sorted = False
            for bin_name, keywords in bins.items():
                if any(keyword in file_name for keyword in keywords):
                    bin_folder = os.path.join(output_folder, bin_name)
                    if not os.path.exists(bin_folder):
                        os.makedirs(bin_folder)
                    shutil.move(file_path, bin_folder)
                    print(f"moved {file_path} to {bin_folder}")
                    sorted = True
                    break
            if not sorted:
                other_folder = os.path.join(output_folder, "_misc")
                if not os.path.exists(other_folder):
                    os.makedirs(other_folder)
                shutil.move(file_path, other_folder)
                print(f"moved {file_path} to {other_folder}")
    
    if is_folder_empty(audio_folder):
        os.rmdir(audio_folder)

def is_folder_empty(folder_path):
    return not any(os.scandir(folder_path))  
      
def main():
    start_time = time.time()
    
    parent_folder = r'/Users/josiahkorzan/Desktop/Programming/Projects/Sample Organization Project/test folders/Samples'
    output_folder = r'/Users/josiahkorzan/Desktop/Programming/Projects/Sample Organization Project/test folders/Samples_Sorted'
    copy_files_to_output(parent_folder, output_folder)
    sort_audio_files_by_file_type(output_folder)
    
    bins_main = {
        "loops": ["loop", "bpm", "key"],
        "kick": ["kik", "kicks", "kick"],
        "snare": ["snr", "snares", "snare"],
        "hi-hat": ["hihat", "hi hats", "hat", "hh", " hi ", "open hi", "oh"],
        "tom": ["tom"],
        "wood": ["wood"],
        "cymbal": ["cymbal", "cymb", "cym", "cy"],
        "crash": ["crash"],
        "ride": ["ride"],
        "maracas": ["maraca"],
        "triangle": ["triangle"],
        "slap": ["slap"],
        "snap": ["snap"],
        "clap": ["clap", "clp"],
        "rimshot": ["rimshot", "rim"],
        "shaker": ["shaker", "shkr"],
        "tambourine": ["tambourine", "tamborine", "tamb"],
        "block": ["block"],
        "whistle": ["whistle"],
        "clave": ["clave"],
        "conga": ["conga", "cong"],
        "bongo": ["bongo"],
        "fx": ["fx", "sfx"],
        "percussion": ["per", "perc", "percussion", "procussion"],
        "bass": ["bassline", "bass", "bass hit"],
        "synth lead": ["synth lead", "synth"],
        "pad": ["pad"],
        "string": ["string"],
        "cowbell": ["cowbell", "cow"],
        "piano": ["piano"],
        "guitar": ["guitar"],
        "brass": ["brass "],
        "woodwind": ["woodwind"],
        "vocals": ["vocal chops", "vocalchop", "vocal", "voice"],
        "riser": ["riser"],
        "impact": ["impact"],
        "sweep": ["sweep"],
        "atmosphere": ["atmosphere", "background", "soundscapes"],
        "foley": ["foley"],
        "glitch": ["glitch"],
        "noise": ["noise"],
        "808": [" 808 "]
    }
    bins_loops = {
        "drum loops": ["drum"],
        "samples": ["sample"],
    }
    sort_audio_files_by_name(output_folder, output_folder, bins_main)
    sort_audio_files_by_name(f"{output_folder}/loops", f"{output_folder}/loops", bins_loops)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Done. Time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()