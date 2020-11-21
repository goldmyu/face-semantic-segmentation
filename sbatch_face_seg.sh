#!/bin/bash

##################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like ##SBATCH
##################

#SBATCH --partition short						### specify partition name where to run a job. debug: 2 hours limit; short: 7 days limit
#SBATCH --time 7-00:00:00			            ### limit the time of job running, partition limit can override this. Format: D-H:MM:SS
#SBATCH --job-name face_seg			            ### name of the job
#SBATCH --output generated_face_seg/output_logs/test_face_seg-%J.out	### output log for running job - %J for job number
#SBATCH --mail-user=goldmyu@post.bgu.ac.il	    ### user email for sending job status
#SBATCH --mail-type=FAIL,END						### conditions when to send the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --gres=gpu:1							### number of GPUs, ask for more than 1 only if you can parallelize your code for multi GPU
##SBATCH --mem=32G
##SBATCH --cpus-per-task=12



### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n"

### Start you code below ####
module load anaconda					### load anaconda module (must present when working with conda environments)
source activate nasutils				### activating environment, environment must be configured before running the job
#srun --mem=24G jupyter lab				### execute jupyter lab command – replace with your own command
										### (e.g. “srun --mem=24G python my.py my_arg”.
										### You may use multiple srun lines, they are the job steps.
										### --mem - memory to allocate: use 24G x number for each allocated GPUs (24G * nGPU)

python src/face_segmentation/face_segmentation.py