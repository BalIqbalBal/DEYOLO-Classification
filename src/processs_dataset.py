from utils.dataset_treatment import delete_duplicate_files_randomly, delete_duplicate_images, organize_files_by_label, randomly_delete_files, rename_rgb_files, rename_thermal_files, sync_pipeline, crop_face_yolov5, sync_directories

#print("Crop Face")
#crop_face_dlib("dataset/dataset_thermal", "dataset/dataset_face_thermal")
#crop_face_mtcnn("dataset/dataset_thermal", "dataset/dataset_face_thermal")

#print("Fill Missing Image")
#sync_pipeline("dataset/datasets_face_thermal", "dataset/datasets_face_rgb")
#sync_directories("dataset/datasets_face_thermal", "dataset/datasets_face_rgb")

#sync_pipeline("dataset/Dataset_Thermal", "dataset/datasets_zip_RGB_Face_Only")
#randomly_delete_files("dataset", 0.75)
#crop_face_custom_haar("dataset/Dataset_Thermal", "dataset/Crop_Dataset_Thermal", "face.xml")

#crop_face_yolov5("dataset/Dataset_Thermal", "dataset/Crop_Dataset_Thermal")

#print("New dataset format")
#organize_files_by_label("dataset/Crop_Dataset_Thermal", "dataset/datasets_zip_RGB_Face_Only", "dataset/formatted_dataset/thermal", "dataset/formatted_dataset/rgb")

#delete_duplicate_images("dataset/formatted_dataset", "dataset/reduce_formatted_dataset")
delete_duplicate_files_randomly("dataset/formatted_dataset/rgb/label0", "dataset/formatted_dataset/thermal/label0", fraction=0.5)