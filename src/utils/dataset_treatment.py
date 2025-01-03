import os
import random
import shutil
#from facenet_pytorch import MTCNN
from PIL import Image
import cv2
import numpy as np

def resize_image(image):
    width = int(image.shape[1])
    height = int(image.shape[0])
    dim = (width, height)
    # Resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def crop_face_HOG(source_dir, target_dir):
    # Inisialisasi detektor wajah HOG dari OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for root, _, files in os.walk(source_dir):
        for file in files:
            # Path lengkap file
            source_path = os.path.join(root, file)
            
            # Path tujuan dengan mengganti parent directory
            relative_path = os.path.relpath(source_path, source_dir)
            target_path = os.path.join(target_dir, relative_path)
            
            # Pastikan direktori tujuan ada
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            try:
                # Buka gambar menggunakan OpenCV (untuk thermal image)
                img = cv2.imread(source_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Konversi ke grayscale
                
                # Deteksi wajah
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(faces) > 0:
                    for i, (x, y, w, h) in enumerate(faces):
                        # Crop wajah berdasarkan koordinat deteksi
                        cropped_face = img[y:y+h, x:x+w]
                        
                        # Convert to RGB for saving with PIL
                        cropped_face_pil = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
                        
                        # Nama file baru untuk wajah
                        face_target_path = target_path.replace(".jpg", f"_face{i+1}.jpg")
                        
                        # Simpan wajah
                        cropped_face_pil.save(face_target_path)
                else:
                    print(f"No face detected in {source_path}")
            
            except Exception as e:
                print(f"Error processing {source_path}: {e}")

"""
def crop_face_mtcnn(source_dir, target_dir):
    # Inisialisasi MTCNN
    mtcnn = MTCNN(keep_all=True)
    
    # List of valid image extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    for root, _, files in os.walk(source_dir):
        for file in files:
            # Get the file extension
            _, ext = os.path.splitext(file)
            
            # Check if the file is an image based on the extension
            if ext.lower() not in valid_extensions:
                continue  # Skip non-image files
            
            # Path lengkap file
            source_path = os.path.join(root, file)
            
            # Path tujuan dengan mengganti parent directory
            relative_path = os.path.relpath(source_path, source_dir)
            target_path = os.path.join(target_dir, relative_path)
            
            # Pastikan direktori tujuan ada
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            try:
                # Buka gambar
                img = Image.open(source_path).convert("RGB")
                
                # Deteksi wajah
                boxes, _ = mtcnn.detect(img)
                
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Crop wajah berdasarkan bounding box
                        cropped_face = img.crop(box)
                        
                        # Nama file baru untuk wajah
                        face_target_path = target_path.replace(".jpg", f"_face{i+1}.jpg")
                        
                        # Simpan wajah
                        cropped_face.save(face_target_path)
                else:
                    # Skip if no face is detected
                    pass
            
            except Exception as e:
                print(f"Error processing {source_path}: {e}")
"""

def get_file_count(directory):
    """Mendapatkan jumlah file dalam direktori."""
    if not os.path.exists(directory):
        return 0
    return len([file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))])

def get_next_file_number(directory):
    """Mengambil angka berikutnya berdasarkan nama file di direktori."""
    files = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
    numbers = []
    for file in files:
        name, _ = os.path.splitext(file)
        if name.isdigit():
            numbers.append(int(name))
    return max(numbers, default=0) + 1

def rename_rgb_files(directory):
    """Mengubah nama file RGB dari format seperti RGB-11-51-02-0121 menjadi angka urut."""
    files = sorted([file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))])
    for idx, file in enumerate(files, start=1):
        _, ext = os.path.splitext(file)
        new_name = f"{idx}{ext}"
        old_path = os.path.join(directory, file)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)

def rename_thermal_files(directory):
    """Mengubah nama file Thermal dari format seperti T145 menjadi angka urut."""
    files = sorted([file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))])
    for idx, file in enumerate(files, start=1):
        _, ext = os.path.splitext(file)
        new_name = f"{idx}{ext}"
        old_path = os.path.join(directory, file)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)

def duplicate_random_files(src_dir, target_dir, difference):
    """Menduplikasi file secara acak di target_dir agar sesuai jumlah dengan src_dir."""
    files = [file for file in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, file))]
    if not files:
        raise ValueError(f"Tidak ada file untuk diduplikasi di {target_dir}")

    for _ in range(difference):
        file_to_duplicate = random.choice(files)
        source_path = os.path.join(target_dir, file_to_duplicate)
        new_file_number = get_next_file_number(target_dir)
        new_file_name = f"{new_file_number}.jpg"  # Sesuaikan ekstensi jika perlu
        target_path = os.path.join(target_dir, new_file_name)
        shutil.copy(source_path, target_path)

def sync_directories(base_dir1, base_dir2):
    """Sinkronisasi file di folder 'label' antara dataset thermal dan RGB."""
    
    for part1 in os.listdir(base_dir1):
        part_dir1 = os.path.join(base_dir1, part1)
        part_dir2 = os.path.join(base_dir2, part1.replace("T_Annotated", "RGB_FaceOnly_Annotated"))

        if not os.path.isdir(part_dir1) or not os.path.isdir(part_dir2):
            print(f"Skipping: {part_dir1} or {part_dir2} does not exist.")
            continue

        for sub in os.listdir(part_dir1):
            sub_dir1 = os.path.join(part_dir1, sub)
            sub_dir2 = os.path.join(part_dir2, sub)

            if not os.path.isdir(sub_dir1) or not os.path.isdir(sub_dir2):
                print(f"Skipping: {sub_dir1} or {sub_dir2} does not exist.")
                continue

            for trial in os.listdir(sub_dir1):
                trial_dir1 = os.path.join(sub_dir1, trial)
                trial_dir2 = os.path.join(sub_dir2, trial)

                if not os.path.isdir(trial_dir1) or not os.path.isdir(trial_dir2):
                    print(f"Skipping: {trial_dir1} or {trial_dir2} does not exist.")
                    continue

                for label in os.listdir(trial_dir1):
                    label_dir1 = os.path.join(trial_dir1, label)
                    label_dir2 = os.path.join(trial_dir2, label)

                    if not os.path.isdir(label_dir1) or not os.path.isdir(label_dir2):
                        print(f"Skipping: {label_dir1} or {label_dir2} does not exist.")
                        continue

                    count1 = get_file_count(label_dir1)
                    count2 = get_file_count(label_dir2)

                    print(f"Checking {label_dir1} ({count1} files) and {label_dir2} ({count2} files)")

                    if count1 > count2:
                        difference = count1 - count2
                        print(f"Duplicating {difference} files from {label_dir1} to {label_dir2}")
                        duplicate_random_files(label_dir1, label_dir2, difference)
                    elif count2 > count1:
                        difference = count2 - count1
                        print(f"Duplicating {difference} files from {label_dir2} to {label_dir1}")
                        duplicate_random_files(label_dir2, label_dir1, difference)
                        
def sync_pipeline(base_dir_thermal, base_dir_rgb):
    """Pipeline untuk merename file dan menyamakan jumlah file di direktori pasangan berdasarkan sub, trial, dan label."""
    for part_thermal in os.listdir(base_dir_thermal):
        part_dir_thermal = os.path.join(base_dir_thermal, part_thermal)

        if not os.path.isdir(part_dir_thermal):
            continue

        for sub_thermal in os.listdir(part_dir_thermal):
            sub_dir_thermal = os.path.join(part_dir_thermal, sub_thermal)

            if not os.path.isdir(sub_dir_thermal):
                continue

            # Cari pasangan sub yang sama di direktori RGB
            for part_rgb in os.listdir(base_dir_rgb):
                part_dir_rgb = os.path.join(base_dir_rgb, part_rgb)
                sub_dir_rgb = os.path.join(part_dir_rgb, sub_thermal)

                if not os.path.isdir(sub_dir_rgb):
                    continue

                for trial_thermal in os.listdir(sub_dir_thermal):
                    trial_dir_thermal = os.path.join(sub_dir_thermal, trial_thermal)

                    if not os.path.isdir(trial_dir_thermal):
                        continue

                    # Cari pasangan trial yang sama di direktori RGB
                    trial_dir_rgb = os.path.join(sub_dir_rgb, trial_thermal)

                    if not os.path.isdir(trial_dir_rgb):
                        continue

                    for label_thermal in os.listdir(trial_dir_thermal):
                        label_dir_thermal = os.path.join(trial_dir_thermal, label_thermal)

                        if not os.path.isdir(label_dir_thermal):
                            continue

                        # Cari pasangan label yang sama di direktori RGB
                        label_dir_rgb = os.path.join(trial_dir_rgb, label_thermal)

                        if not os.path.isdir(label_dir_rgb):
                            print(f"Peringatan: Pasangan untuk {label_dir_thermal} tidak ditemukan di {trial_dir_rgb}")
                            continue

                        print(f"Renaming files in {label_dir_thermal} and {label_dir_rgb}")
                        #rename_thermal_files(label_dir_thermal)
                        rename_rgb_files(label_dir_rgb)

    print("Synchronizing directories...")
    sync_directories(base_dir_thermal, base_dir_rgb)

def organize_files_by_label(source_dir_thermal, source_dir_rgb, target_dir_thermal, target_dir_rgb):
    """
    Menyalin file Thermal dan RGB ke direktori baru berdasarkan label 0-5 dengan format nama yang sesuai.
    Args:
        source_dir_thermal (str): Direktori asal Thermal.
        source_dir_rgb (str): Direktori asal RGB.
        target_dir_thermal (str): Direktori target Thermal.
        target_dir_rgb (str): Direktori target RGB.
    """
    # Buat direktori target dan subdirektorinya
    for target_dir in [target_dir_thermal, target_dir_rgb]:
        os.makedirs(target_dir, exist_ok=True)
        for i in range(6):  # label0 hingga label5
            os.makedirs(os.path.join(target_dir, f"label{i}"), exist_ok=True)

    # Fungsi bantu untuk memproses file dalam satu set direktori
    def process_files(source_dir, target_dir, mode):
        for part in os.listdir(source_dir):
            part_dir = os.path.join(source_dir, part)
            if not os.path.isdir(part_dir):
                continue

            for sub in os.listdir(part_dir):
                sub_dir = os.path.join(part_dir, sub)
                if not os.path.isdir(sub_dir):
                    continue

                for trial in os.listdir(sub_dir):
                    trial_dir = os.path.join(sub_dir, trial)
                    if not os.path.isdir(trial_dir):
                        continue

                    for label in os.listdir(trial_dir):
                        label_dir = os.path.join(trial_dir, label)
                        if not os.path.isdir(label_dir):
                            continue

                        # Salin file dengan format nama baru
                        for idx, file in enumerate(os.listdir(label_dir), start=1):
                            old_path = os.path.join(label_dir, file)
                            if os.path.isfile(old_path):
                                # Format nama file baru
                                new_file_name = f"{sub}_{trial}_{idx}"
                                if mode == "thermal":
                                    new_file_name = f"{new_file_name}"
                                new_file_name += os.path.splitext(file)[1]
                                new_path = os.path.join(target_dir, label, new_file_name)

                                shutil.copy(old_path, new_path)

    # Proses Thermal dan RGB
    process_files(source_dir_thermal, target_dir_thermal, "thermal")
    process_files(source_dir_rgb, target_dir_rgb, "rgb")

    print(f"File berhasil disalin ke {target_dir_thermal} dan {target_dir_rgb}")


def sync_folders(rgb_root, thermal_root):
    """
    Synchronize files between RGB and thermal folders by removing files that don't exist in both directories.
    
    Args:
        rgb_root (str): Path to the RGB root folder
        thermal_root (str): Path to the thermal root folder
    """
    # Get all label subdirectories
    labels = [d for d in os.listdir(rgb_root) if os.path.isdir(os.path.join(rgb_root, d))]
    
    files_to_remove = []
    
    # Iterate through each label directory
    for label in labels:
        rgb_label_path = os.path.join(rgb_root, label)
        thermal_label_path = os.path.join(thermal_root, label)
        
        # Skip if the label doesn't exist in thermal folder
        if not os.path.exists(thermal_label_path):
            continue
        
        # Get files in both directories
        rgb_files = set(os.listdir(rgb_label_path))
        thermal_files = set(os.listdir(thermal_label_path))
        
        # Find files that exist in only one directory
        rgb_only = rgb_files - thermal_files
        thermal_only = thermal_files - rgb_files
        
        # Add files to removal list
        for file in rgb_only:
            files_to_remove.append(os.path.join(rgb_label_path, file))
        for file in thermal_only:
            files_to_remove.append(os.path.join(thermal_label_path, file))
    
    # Show files that will be removed
    if files_to_remove:
        print("The following files will be removed:")
        for file in files_to_remove:
            print(f"- {file}")
        
        # Ask for confirmation
        response = input("\nDo you want to proceed with deletion? (yes/no): ")
        if response.lower() == 'yes':
            # Remove the files
            for file in files_to_remove:
                os.remove(file)
            print("\nFiles have been removed successfully.")
        else:
            print("\nOperation cancelled.")
    else:
        print("No files need to be removed. The directories are already synchronized.")



