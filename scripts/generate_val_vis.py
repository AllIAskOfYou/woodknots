import cv2
import hydra
from hydra.core.hydra_config import HydraConfig
import lightning as L
import numpy as np
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import argparse
from monai.inferers import sliding_window_inference
from fpdf import FPDF
import tqdm

from mltools.seg.dataset import get_dataset_by_config, SegDataset
from mltools.seg.schema import TrainConfig
from mltools.seg.transform import (
    get_preprocessing_transform,
    get_train_transform,
    get_val_transform,
    INFERENCE_TRANSFORM,
)
from mltools.seg.module import SegModule
from mltools.seg.module.seg_module import denormalize_image
from mltools.utils.data import get_metadata, get_fold

def generate_visualizations():
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg: TrainConfig = hydra.compose(
            config_name="config",
        )

    if cfg.dataset.resolution < cfg.augmentation.resolution:
        raise ValueError(
            f"Dataset resolution ({cfg.dataset.resolution}) must be >= augmentation resolution ({cfg.augmentation.resolution})"
        )
    
    vis_dir = Path("visualizations")
    vis_dir.mkdir(parents=True, exist_ok=True)
    (vis_dir / "images").mkdir(parents=True, exist_ok=True)
    (vis_dir / "images_vis").mkdir(parents=True, exist_ok=True)

    root_path = get_dataset_by_config(cfg.dataset)
    metadata = get_metadata(root_path)
    channel_index = metadata["channel_names"].index(cfg.class_name)
    train_cases, val_cases = get_fold(root_path, fold=cfg.fold, k=5)
    preprocessing = get_preprocessing_transform(cfg.preprocessing)
    # val_transform = INFERENCE_TRANSFORM
    val_transform = get_val_transform(cfg.augmentation)

    val_dataset = SegDataset(
        root_path,
        channel_index=channel_index,
        cases=val_cases,
        preprocessing=preprocessing,
        transform=val_transform,
        class_groups=cfg.class_groups,
        cache_amount=1.0
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    print(f"Val Dataset: {len(val_dataset)} samples")

    module = SegModule(cfg.module)
    ckpt_path = "experiments/light/2026-06-15/21-08-13/checkpoints/best-model-epoch=33-val_dice=0.8271.ckpt"
    # ckpt_path = "experiments/light/2026-06-15/21-08-13/checkpoints/last.ckpt"
    module = SegModule.load_from_checkpoint(ckpt_path, weights_only=False)
    module.cfg = cfg.module

    module.eval()
    for i, (image, mask) in tqdm.tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        # rotate the image and mask by 90 degrees clockwise if the image is taller than it is wide
        if image.shape[2] > image.shape[3]:
            image = torch.rot90(image, k=1, dims=(2, 3))
            mask = torch.rot90(mask, k=1, dims=(2, 3))
        image = image.to(module.device)
        with torch.no_grad():
            res = cfg.augmentation.resolution
            outputs = sliding_window_inference(
                image.to(module.device),
                roi_size=(res, res),
                sw_batch_size=4,
                predictor=module.model,
                overlap=0.5,
                mode="gaussian",
            )
            outputs = torch.sigmoid(outputs)
        outputs = outputs.detach().cpu().numpy()
        # paint contours on the original image using true mask
        image_denorm = (denormalize_image(image[0]) * 255).astype(np.uint8)
        image_denorm = cv2.cvtColor(image_denorm, cv2.COLOR_RGB2BGR)
        # overlay the image with predicted mask in blue color
        image_vis = image_denorm.copy()
        transparency = outputs[0, 0, :, :, np.newaxis]
        image_vis = image_vis.astype(np.float32) * (1 - transparency) + np.array([255, 0, 0], dtype=np.float32) * transparency
        image_vis = image_vis.astype(np.uint8)
        mask_long = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
                mask_long[0, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
        cv2.drawContours(image_vis, contours, -1, (0, 0, 255), 2)
        for contour in contours:
            # add text on top of the contour with the value of the mask
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(
                    image_vis,
                    f"{mask_long[0, 0, cY, cX] / 255:.1f}",
                    (cX - 20, cY),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
        image_vis = cv2.resize(image_vis, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imwrite(vis_dir / "images_vis" / f"val_image_{i}.jpg", image_vis)
        cv2.imwrite(vis_dir / "images" / f"val_image_{i}.jpg", image_denorm)

def generate_pdf():
    # compute image height based on number of rows
    vis_dir = Path("visualizations")
    images = sorted(list((vis_dir / "images").glob("*.jpg")), key=lambda x: int(x.stem.split("_")[-1]))
    images_vis = sorted(list((vis_dir / "images_vis").glob("*.jpg")), key=lambda x: int(x.stem.split("_")[-1]))
    a4_width_mm = 210
    a4_height_mm = 297
    margin_mm = 5
    between_margin_mm = 2
    num_cols = 2
    usable_width_mm = a4_width_mm - 2 * margin_mm - (num_cols - 1) * between_margin_mm
    image_width_mm = usable_width_mm / num_cols
    image_shape = cv2.imread(str(images[0])).shape
    image_ratio = image_shape[0] / image_shape[1]
    image_height_mm = image_width_mm * image_ratio
    num_rows = int((a4_height_mm - 2 * margin_mm + between_margin_mm) / (image_height_mm + between_margin_mm))

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    # Display the title of the PDF

    # --- Title ---
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Vizualizacija validacijske mnozice", ln=True, align="C")
    pdf.ln(5)

    # --- Subtitles (Leva slika / Desna slika) ---
    # Assuming printable width is ~210 - 20 (margins) = 190 mm
    # Let's use two columns of 90 mm width with a 10 mm gap in between
    col_width = image_width_mm
    gap = between_margin_mm

    pdf.set_font("Arial", "", 11)
    pdf.cell(col_width, margin_mm, "LEVA SLIKA", align="C")
    pdf.cell(gap, margin_mm, "")  # Spacer
    pdf.cell(col_width, margin_mm, "DESNA SLIKA", align="C", ln=True)

    # --- Images / Boxes setup ---
    # Given dimensions for the image boxes

    # Save current X and Y position before drawing boxes
    start_x = margin_mm
    start_y = pdf.get_y()

    # Draw Left Box Outline
    pdf.rect(start_x, start_y, col_width, image_height_mm)

    # Draw Right Box Outline
    right_box_x = start_x + col_width + gap
    pdf.rect(right_box_x, start_y, col_width, image_height_mm)

    # --- Content inside Left Box ---
    # Center "Izvirna slika" roughly in the middle of the left box
    pdf.set_xy(start_x, start_y + (image_height_mm / 2) - 4)
    pdf.set_font("Arial", "", 14)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(col_width, 8, "Izvirna slika", align="C")

    # --- Content inside Right Box ---
    pdf.set_xy(right_box_x + 5, start_y + 5)  # Add a small padding inside the box
    pdf.set_font("Arial", "B", 11)

    # Rdece header
    pdf.set_text_color(220, 50, 50)
    pdf.cell(col_width - 10, 6, "Rdece:", ln=True)

    # Rdece description
    pdf.set_x(right_box_x + 5)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(col_width - 10, 6, "- oznaka anotatorja", ln=True)

    pdf.ln(4)

    # Modro header
    pdf.set_x(right_box_x + 5)
    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(50, 50, 220)
    pdf.cell(col_width - 10, 6, "Modro:", ln=True)

    # Modro description
    pdf.set_x(right_box_x + 5)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(col_width - 10, 6, "- napovedana oznaka", ln=True)

    # Reset text color back to default black for future elements
    pdf.set_text_color(0, 0, 0)

    # Move cursor below the boxes for subsequent content
    pdf.set_xy(start_x, start_y + image_height_mm + 10)
    pdf.add_page()

    if len(images) == 0:
        print("No images found in visualizations directory. Please run the script without --only-pdf to generate visualizations first.")
        return
    num_pages = (len(images) + num_rows - 1) // num_rows
    for page in range(num_pages):
        # if page > 0:
        #     break
        if page > 0:
            pdf.add_page()
        for row in range(num_rows):
            # on the left put the original image, on the right put the visualization
            for col in range(num_cols):
                idx = page * num_rows + row
                if idx >= len(images):
                    break
                x_mm = margin_mm + col * (image_width_mm + between_margin_mm)
                y_mm = margin_mm + row * (image_height_mm + between_margin_mm)
                if col == 0:
                    pdf.image(str(images[idx]), x=x_mm, y=y_mm, w=image_width_mm, h=image_height_mm)
                else:
                    pdf.image(str(images_vis[idx]), x=x_mm, y=y_mm, w=image_width_mm, h=image_height_mm)
    out_path = vis_dir / "validation_visualizations.pdf"
    print(f"Saving PDF to {out_path}")
    pdf.output(str(out_path))

def main():
    argparser = argparse.ArgumentParser(description="Generate visualizations for validation dataset")
    argparser.add_argument("--only-pdf", action="store_true", help="Only generate PDF visualizations")
    args = argparser.parse_args()
    if not args.only_pdf:
        generate_visualizations()
    else:
        print("Skipping visualization generation, only generating PDF")
    generate_pdf()



if __name__ == "__main__":
    main()
