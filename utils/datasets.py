import os
import shutil
import pandas as pd

def organize_datasets(train_csv, valid_csv, test_csv, output_base_dir):
    """
    Organize images into train/valid/test folders with trial and label subdirectories.
    """
    # Create output directories
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Function to copy files with trial and label structure
    def copy_files(csv_path, dest_folder):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            source_filepath = row['filepath']
            trial = row['trial']
            label = row['label']
            
            # Create destination path with trial and label structure
            dest_dir = os.path.join(output_base_dir, dest_folder, trial, label)
            os.makedirs(dest_dir, exist_ok=True)
            
            # Copy file to new destination
            filename = os.path.basename(source_filepath)
            dest_path = os.path.join(dest_dir, filename)
            
            if os.path.exists(source_filepath):
                try:
                    shutil.copy2(source_filepath, dest_path)
                except Exception as e:
                    print(f"Error copying {source_filepath}: {e}")
            else:
                print(f"Warning: File not found - {source_filepath}")
    
    # Organize datasets
    print("Copying training files...")
    copy_files(train_csv, 'train')
    
    print("Copying validation files...")
    copy_files(valid_csv, 'valid')
    
    print("Copying test files...")
    copy_files(test_csv, 'test')
    
    print("Dataset organization complete!")

# Example usage
if __name__ == "__main__":
    # Specify paths to your CSV files
    train_csv = 'train.csv'
    valid_csv = 'valid.csv'
    test_csv = 'test.csv'
    
    # Base directory where organized datasets will be created
    output_base_dir = './organized_datasets'
    
    # Call the function to organize datasets
    organize_datasets(train_csv, valid_csv, test_csv, output_base_dir)