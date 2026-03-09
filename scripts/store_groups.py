import json
import os
from mltools import METADATA_DIR

ORIGINAL_DIR = "/mnt/d/matics/logs/raw"


def get_images_by_tree():
    trees = os.listdir(ORIGINAL_DIR)
    images_by_tree = list()
    for i, tree in enumerate(trees):
        tree_path = os.path.join(ORIGINAL_DIR, tree)
        images = os.listdir(tree_path)
        images_by_tree.append([".".join(image.split(".")[:-1]) for image in images])
    return images_by_tree


def main():
    images_by_tree = get_images_by_tree()
    with open(os.path.join(METADATA_DIR, "groups.json"), "w") as f:
        json.dump(images_by_tree, f, indent=4)


if __name__ == "__main__":
    main()
