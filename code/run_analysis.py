import pickle
import psutil
import numpy as np
import multiprocessing
from pathlib import Path
from scipy.stats.qmc import LatinHypercube
from TippingFPE import run_fpe_analysis
from tqdm import tqdm
from datetime import datetime

#  ----- SIMULATION PROPERTIES

d = 6       # number of parameter dimensions
N = 500     # number of samples
upper =  1  # upper bound of parameter values
lower = -1  # lower bound of parameter values

out = "./out" 

# ------

# make directory if it does not exist
Path(out).mkdir(parents=True, exist_ok=True)

# generate LH samples
samples = LatinHypercube(d).random(N)
samples = (upper - lower)*samples + lower

def run_and_store(sample):

    # prepare coupling matrix    
    coupling_matrix = np.zeros((3, 3))
    coupling_matrix[0, 1] = sample[0]
    coupling_matrix[1, 0] = sample[1]
    coupling_matrix[0, 2] = sample[2]
    coupling_matrix[2, 0] = sample[3]
    coupling_matrix[1, 2] = sample[4]
    coupling_matrix[2, 1] = sample[5]

    # run analysis
    summary = run_fpe_analysis(coupling_matrix, grid_extent=[[-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5]], grid_shape=[150, 150, 150], noise_level=0.1, p0="all", max_runtime_seconds=5*60)

    # store results
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"{out}/{now}.pkl", "wb") as file:
        pickle.dump([coupling_matrix, *summary], file)

    # clear unused memory
    del coupling_matrix, summary


if __name__ == "__main__":

    print(psutil.cpu_count())

    pool = multiprocessing.Pool(processes=10, maxtasksperchild=10)
    for _ in tqdm(pool.imap_unordered(run_and_store, samples), total=len(samples)):
        pass