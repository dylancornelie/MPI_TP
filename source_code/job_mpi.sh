#!/bin/bash

#SBATCH --job-name=MPI_TP
#SBATCH --output=N10n10.txt
#SBATCH -N 2 # number of nodes
#SBATCH -n 6  # number of cores/processes
#SBATCH --distribution=cyclic
#SBATCH --exclusive
#SBATCH --partition=nodo.q

module load openmpi/3.0.0
mpicc *.c -o test  -lmpi -lm -Ofast -std=c99
perf stat mpirun ./test
