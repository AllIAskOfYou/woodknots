import pandas as pd

def main():
    df_nnunet = pd.read_csv("article/conf_sweep/nnunet_stored.csv")
    df_yolo = pd.read_csv("article/conf_sweep/yolo_stored.csv")
    df_seg = pd.read_csv("article/conf_sweep/seg_stored.csv")

    # select only dice_fg_pooled and mAP50, rename dice_fg_pooled to Dice
    df_nnunet = df_nnunet[["fold", "dice_fg_pooled", "mAP50"]].rename(columns={"dice_fg_pooled": "Dice"})
    df_yolo = df_yolo[["fold", "dice_fg_pooled", "mAP50"]].rename(columns={"dice_fg_pooled": "Dice"})
    df_seg = df_seg[["fold", "dice_fg_pooled", "mAP50"]].rename(columns={"dice_fg_pooled": "Dice"})

    # aggregate by fold and compute mean and std for Dice and mAP50
    df_nnunet = df_nnunet.groupby("fold").agg({"Dice": ["mean", "std"], "mAP50": ["mean", "std"]})
    df_yolo = df_yolo.groupby("fold").agg({"Dice": ["mean", "std"], "mAP50": ["mean", "std"]})
    df_seg = df_seg.groupby("fold").agg({"Dice": ["mean", "std"], "mAP50": ["mean", "std"]})

    # print the results to a file
    with open("article/folds_report.txt", "w") as f:
        f.write("nnUNet:\n")
        f.write(df_nnunet.to_string())
        f.write("\n\nYOLO:\n")
        f.write(df_yolo.to_string())
        f.write("\n\nCustom:\n")
        f.write(df_seg.to_string())
    print(df_nnunet)
    print(df_yolo)
    print(df_seg)
    print("Results written to article/folds_report.txt")

if __name__ == "__main__":
    main()