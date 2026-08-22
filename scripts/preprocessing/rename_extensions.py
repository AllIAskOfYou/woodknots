import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename image extensions to lowercase")
    parser.add_argument("directory", type=str, help="Directory containing images")
    args = parser.parse_args()
    for filename in os.listdir(args.directory):
        if filename.lower().endswith((".jpg", ".jpeg")):
            old_path = os.path.join(args.directory, filename)
            new_filename = filename.rsplit(".", 1)[0] + ".jpg"
            new_path = os.path.join(args.directory, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")