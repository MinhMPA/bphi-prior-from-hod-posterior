source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
# create Abacus mocks
#calculator="srun -N 1 -n 1 -c 32 -C cpu --mem=70G -t 01:00:00 --qos interactive --account desi "
#${calculator} python AbacusHOD_lowS8.py

# compute the bias and bphi
#calculator="srun -N 1 -n 1 -c 32 -C cpu --mem=70G -t 01:00:00 --qos shared_interactive --account desi "
OMP_NUM_THREADS=128
${calculator} python lowS8_comparison.py

