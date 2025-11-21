import os
import shutil
import random

def distribute_images(labeled_path, unlabeled_path, output_root):
    # Verify paths exist
    if not os.path.isdir(labeled_path):
        raise FileNotFoundError(f"Labeled path does not exist: {labeled_path}")
    if not os.path.isdir(unlabeled_path):
        raise FileNotFoundError(f"Unlabeled path does not exist: {unlabeled_path}")

    people = ["David", "Viki", "Mihajlo"]

    # Create output folders
    for person in people:
        os.makedirs(os.path.join(output_root, person, "labeled"), exist_ok=True)
        os.makedirs(os.path.join(output_root, person, "unlabeled"), exist_ok=True)

    # Helper: split list into 3 equal parts
    def split_equal(file_list):
        random.shuffle(file_list)
        chunk = len(file_list) // 3
        return {
            "David": file_list[:chunk],
            "Viki": file_list[chunk:2*chunk],
            "Mihajlo": file_list[2*chunk:]
        }

    # Get labeled and unlabeled files
    labeled_files = [
        os.path.join(labeled_path, f)
        for f in os.listdir(labeled_path)
        if os.path.isfile(os.path.join(labeled_path, f))
    ]

    unlabeled_files = [
        os.path.join(unlabeled_path, f)
        for f in os.listdir(unlabeled_path)
        if os.path.isfile(os.path.join(unlabeled_path, f))
    ]

    # Split
    labeled_split = split_equal(labeled_files)
    unlabeled_split = split_equal(unlabeled_files)

    # Copy files
    for person in people:
        for file in labeled_split[person]:
            shutil.copy(file, os.path.join(output_root, person, "labeled"))
        for file in unlabeled_split[person]:
            shutil.copy(file, os.path.join(output_root, person, "unlabeled"))

    print("✔ Distribution complete!")


# ----------------- Run -----------------
if __name__ == "__main__":
    labeled_images_path = r"E:\FAKULTET\Semestar7\Inteligentni informaciksi sistemi\seminarska\FoodID_Dataset"
    unlabeled_images_path = r"E:\FAKULTET\Semestar7\Inteligentni informaciksi sistemi\seminarska\Unlabeled"
    output_path = r"E:\FAKULTET\Semestar7\Inteligentni informaciksi sistemi\seminarska\output_split"

    distribute_images(labeled_images_path, unlabeled_images_path, output_path)
