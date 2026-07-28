import cv2
from pathlib import Path
import tqdm

if __name__ == "__main__":
    input_dir = Path("data/matching/raw")
    res = 512
    output_dir = Path(f"data/matching/resized_{res}")
    output_dir.mkdir(exist_ok=True)

    images = list(input_dir.glob("*.JPG")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    for image_path in tqdm.tqdm(images, desc="Resizing images", total=len(images)):
        image = cv2.imread(str(image_path))
        # resize so the shortest side is res pixels, keeping the aspect ratio
        h, w = image.shape[:2]
        if h < w:
            new_h = res
            new_w = int(w * (res / h))
        else:
            new_w = res
            new_h = int(h * (res / w))
        resized_image = cv2.resize(image, (new_w, new_h))
        cv2.imwrite(str(output_dir / image_path.name), resized_image)