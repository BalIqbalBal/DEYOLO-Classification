import os
import shutil

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

# Example usage
if __name__ == "__main__":
    rgb_root = "dataset/formatted_dataset/rgb"     # Replace with your RGB folder path
    thermal_root = "dataset/formatted_dataset/thermal"  # Replace with your thermal folder path
    
    sync_folders(rgb_root, thermal_root)