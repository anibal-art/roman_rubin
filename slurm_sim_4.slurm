#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --hint=multithread
#SBATCH --mem=250G
#SBATCH --partition=milliways-int
#SBATCH --job-name="set4"
##SBATCH --mail-type=FAIL,END
##SBATCH --mail-user=anibalvarela134@gmail.com

python run_paralelo_set4.py
