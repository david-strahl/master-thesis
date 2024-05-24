import warnings
import numpy as np

# temperature limits for GIS, AMOC, WAIS, AMAZ
T = np.array([(0.8, 3.0), (1.4, 8.0), (1.0, 3.0), (2.0, 6.0)])

# coupling coefficients
S = np.zeros((4,4,2))
S[1, 0] = [ 0,  10] # GIS -> AMOC
S[0, 1] = [-10,  0] # AMOC -> GIS
S[2, 0] = [ 0,  10] # GIS -> WAIS
S[3, 1] = [-4,   4] # AMOC -> AMAZ
S[1, 2] = [-3,   3] # WAIS -> AMOC
S[0, 2] = [ 0,   2] # WAIS -> GIS
S[2, 1] = [ 0, 1.5] # AMOC -> WAIS
S = S/10

def random_temperatures(n=1, squeeze=True):
	t = np.random.uniform(T[:,0], T[:,1], (n, 4))
	return np.squeeze(t) if squeeze else t

def random_couplings(strength=1, links=7):
    S = strength*np.random.uniform(S[:,:,0], S[:,:,1])
    idx = np.array([[0, 2], [0, 1], [1, 0], [1, 2], [2, 0], [2, 1], [3, 1]])
    idx = idx[np.random.choice(7, 7 - links, False)]
    for i, j in idx:
        S[i, j] = 0.0

   #warnings.warn("The coupling seems to be specified in the wrong direction: [i, j] appears to mean a coupling j -> i")

    return S

var_names = ["GIS", "WAIS", "AMOC", "AMAZ"]