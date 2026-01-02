import os
import shutil

source_root = "dataset\Training"  # or your path
target_root = "sd_dataset"
os.makedirs(target_root, exist_ok=True)

caption_map = {
    "glioma": "MRI scan of the brain showing a glioma tumor, btumr_style.",
    "meningioma": "MRI scan of the brain showing a meningioma tumor, btumr_style.",
    "pituitary": "MRI scan of the brain showing a pituitary tumor, btumr_style.",
    "notumor": "MRI scan of the brain with no detectable tumor, btumr_style."
}

idx = 0

for class_name in caption_map.keys():
    class_folder = os.path.join(source_root, class_name)
    for file in os.listdir(class_folder):
        if file.lower().endswith((".jpg",".png",".jpeg")):
            idx += 1
            new_img_name = f"img_{idx:05d}.jpg"
            src = os.path.join(class_folder, file)
            dst = os.path.join(target_root, new_img_name)

            shutil.copy(src, dst)

            # Write caption
            caption_path = os.path.join(target_root, new_img_name.replace(".jpg", ".txt"))
            with open(caption_path, "w") as f:
                f.write(caption_map[class_name])

print("Conversion complete. SD-ready dataset created.")
