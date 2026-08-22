import cv2
from pathlib import Path
import tqdm

if __name__ == "__main__":
    input_dir = Path("data/matching/raw")
    res = 1024
    output_dir = Path(f"data/matching/resized_{res}")
    output_dir.mkdir(exist_ok=True)

    images = list(input_dir.glob("*.JPG")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    for image_path in tqdm.tqdm(images, desc="Resizing images", total=len(images)):
        image = cv2.imread(str(image_path))
        # rotate the image if it's in portrait mode
        if image.shape[0] > image.shape[1]:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        # resize so that the width is res and the height is scaled accordingly
        scale = res / image.shape[1]
        new_height = int(image.shape[0] * scale)
        resized_image = cv2.resize(image, (res, new_height), interpolation=cv2.INTER_LINEAR)
        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), resized_image)