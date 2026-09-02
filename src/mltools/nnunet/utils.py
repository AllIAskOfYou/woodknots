import os


def build_nnunet_env(
    nnunet_raw: str, nnunet_preprocessed: str, nnunet_results: str
) -> dict:
    env = os.environ.copy()
    env["nnUNet_raw"] = nnunet_raw
    env["nnUNet_preprocessed"] = nnunet_preprocessed
    env["nnUNet_results"] = nnunet_results
    return env
