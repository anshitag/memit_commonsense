# Run Instructions:
# wandb sweep sweep_configs/xx.yaml -> copy the SWEEP_ID
# python slurm_wandb_agent.py --swp SWEEP_ID --ptt gypsum-m40 --njobs 2 --nruns 2

import argparse, os, json

parser = argparse.ArgumentParser(description='Run multiple wandb agents via SLURM')
parser.add_argument('--wandb_account', type=str, default="anshitag")
parser.add_argument('--wandb_project', type=str, default="memit_commonsense_edit")
parser.add_argument('--swp', type=str, default=None, help="sweep id")

parser.add_argument('--ptt', type=str, default='gypsum-titanx', help="partition")
parser.add_argument('--exclude', type=str, default=None, help="e.g. node001,node002")
parser.add_argument('--njobs', type=int, default=1, help="number of agents per sweep")
parser.add_argument('--nruns', type=int, default=10, help="number of runs per agent")

parser.add_argument('--slurm_dir', type=str, default="slurm", help="directory to store slurm files in")
parser.add_argument('--srun_filename', type=str, default="srun.sh", help="filename of srun script (to create)")
parser.add_argument('--sbatch_filename', type=str, default="sbatch.sh", help="filename of sbatch script (to create)")
args = parser.parse_args()

os.makedirs(args.slurm_dir, exist_ok=True)
with open(os.path.join(args.slurm_dir, '.gitignore'), 'w') as f:
    f.write('*\n!.gitignore')

if args.swp:
    sweep_ids = [args.swp]
else: # supporting multiple sweeps at a time, usually not needed
    sweep_id_path = "log_sweep_id.json"
    sweep_ids = json.load(open(sweep_id_path, 'r')) if os.path.exists(sweep_id_path) else {}
    sweep_ids = [sweep_id for _, sweep_id in sweep_ids.items()]

for i, sweep_id in enumerate(sweep_ids):
    sweep_dir = os.path.join(args.slurm_dir, sweep_id)
    os.makedirs(sweep_dir, exist_ok=True)
    sbatch_file_path = os.path.join(sweep_dir, args.sbatch_filename)
    srun_file_path = os.path.join(sweep_dir, args.srun_filename)

    with open(sbatch_file_path, 'w') as f:
        f.write(
            "\n".join((
                "#!/bin/bash",
                "#SBATCH --gres=gpu:1",
                f"#SBATCH --partition={args.ptt}",
                f"#SBATCH --exclude={args.exclude}" if args.exclude else "",
                "#SBATCH --mem=50GB",
                f"#SBATCH --array=1-{args.njobs}" if args.njobs > 1 else "",
                f"#SBATCH --job-name={sweep_id}",
                f"#SBATCH --output={sweep_dir}/%A-%a.out" if args.njobs > 1 else f"#SBATCH --output={sweep_dir}/%A.out",
                f"srun {srun_file_path}"
            ))
        )
    os.system(f"chmod +x {sbatch_file_path}")

    with open(srun_file_path, 'w') as f:
        f.write(
            "\n".join((
                "#!/bin/bash",
                "export NO_PROGRESS_BAR=true",
                "hostname",
                f"wandb agent --count {args.nruns} {args.wandb_account}/{args.wandb_project}/{sweep_id}" \
                    if args.nruns else f"wandb agent {args.wandb_account}/{args.wandb_project}/{sweep_id}",
            ))
        )
    os.system(f"chmod +x {srun_file_path}")

    os.system(f"sbatch {sbatch_file_path}")