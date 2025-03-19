import os

def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.startswith("cleaned_"):
            new_filename = filename.replace("cleaned_", "", 1)  # Chỉ thay thế lần đầu tiên
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f'Renamed: {filename} -> {new_filename}')

# Thay thế "your_directory_path" bằng đường dẫn thư mục chứa các file
rename_files("./audio_vad_500/data")
