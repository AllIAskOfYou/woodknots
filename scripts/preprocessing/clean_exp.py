import os

if __name__ == "__main__":
    path = "experiments/light"
    days = sorted(os.listdir(path))
    for day in days:
        day_path = os.path.join(path, day)
        experiments = sorted(os.listdir(day_path))
        for experiment in experiments:
            dir = os.path.join(day_path, experiment)
            if not os.path.isdir(os.path.join(dir, "checkpoints")):
                print(f"Deleting {dir}")
                os.system(f"rm -rf {dir}")