#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --hint=multithread
#SBATCH --mem=250G
#SBATCH --partition=cosmoobs
#SBATCH --job-name="width"
##SBATCH --mail-type=FAIL,END
##SBATCH --mail-user=anibalvarela134@gmail.com

python run_test_width.py
