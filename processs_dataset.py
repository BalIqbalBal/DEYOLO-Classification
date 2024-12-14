from utils.dataset_treatment import sync_pipeline, organize_files_by_label, crop_face_HOG, crop_face_mtcnn

print("Crop Face")
crop_face_mtcnn("dataset/dataset_thermal", "dataset/dataset_face_thermal")

print("Fill Missing Image")
sync_pipeline("dataset/dataset_face_thermal", "dataset/datasets_face_rgb")

print("New dataset format")
organize_files_by_label("dataset/dataset_face_thermal", "dataset/datasets_face_rgb", "dataset/formatted_dataset/thermal", "dataset/formatted_dataset/rgb")