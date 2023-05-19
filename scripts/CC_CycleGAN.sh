#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1	#This line indicates which GPU will be used for the job Example: NVIDIA V100 Volta(32G HBM2 memory)
#SBATCH --cpus-per-task=3		#Indicate how many CPUs will be used	
#SBATCH --mem=100G       		#Memory allocated
#SBATCH --time=7-00:00:00			#Time to run job in form of DD-HH:MM:SS
#SBATCH --output=%N-%j.out		#Indicates the format of the output file
#SBATCH --account= #ComputeCanada account name
#SBATCH --mail-type=ALL			#Will provide email updates regarding the job submitted
#SBATCH --mail-user= #User email

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load python/3.8

SOURCEDIR=.../Image_Synthesis/AllScripts

#Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install tensorflow tensorflow-addons numpy scikit-image scipy matplotlib pandas --no-index

mkdir $SLURM_TMPDIR/data
echo $SLURM_TMPDIR/data
tar xf .../Image_Synthesis/Data_unpaired.tar -C $SLURM_TMPDIR/data	#Unzip data to temporary Directory 

#dir $SLURM_TMPDIR/data/

mkdir .../Image_Synthesis/experiments/experiment10_1_unpaired
mkdir .../Image_Synthesis/experiments/experiment10_1_unpaired/model
mkdir .../Image_Synthesis/experiments/experiment10_1_unpaired/model/training
mkdir .../Image_Synthesis/experiments/experiment10_1_unpaired/model/training_images
mkdir .../Image_Synthesis/experiments/experiment10_1_unpaired/model/checkpoints

# Start training
 python $SOURCEDIR/train_cyclegan.py $SLURM_TMPDIR/data/
