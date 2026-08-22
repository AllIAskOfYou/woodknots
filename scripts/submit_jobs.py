import subprocess
from pathlib import Path

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={exp_name}
#SBATCH --output=logs/{exp_name}_%j.out
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

srun --container-image={container_image} \\
     --container-mounts=/data:/data,{project_dir}:/workspace \\
     --container-workdir=/workspace \\
     python train.py experiment={exp_name}
"""

CONTAINERS = {
    "nnunet": "/scratch/images/nnunet.sqsh",
    "yourframework": "/scratch/images/yourframework.sqsh",
}


def submit(framework: str, project_dir: str):
    exps = get_experiments(framework)
    for exp in exps:
        script = SBATCH_TEMPLATE.format(
            exp_name=exp,
            container_image=CONTAINERS[framework],
            project_dir=project_dir,
        )
        path = Path(f"jobs/{exp}.sbatch")
        path.write_text(script)
        subprocess.run(["sbatch", str(path)], check=True)
