import numpy as np
import scipy as sp
import json
import h5py
import glob 
import os 
from tqdm import tqdm
import argparse 

# Parser
parser = argparse.ArgumentParser(prog="ABPd", description="Compute density field for ABP simulations.")
parser.add_argument(
    "--rootdir", type=str, required=True,
    help="Path to the root directory of the simulation."
)
parser.add_argument(
    "--kernel", type=str, required=True,
    help="Kernel to smooth the field, options: 'gaussian' or 'bump'."
)
parser.add_argument(
    "--rcut", type=float, required=True,
    help="Cutoff radius (in terms of sigma) for the smoothing operation."
)
parser.add_argument(
    "--t0", type=int, default=10,
    help="Wait time (in number of frames) before starting the analysis."
)

args = parser.parse_args()

rootdir = args.rootdir
kernel = args.kernel
rcut = args.rcut
start = args.t0

json_files = glob.glob(os.path.join(rootdir, "*.json"))
with open(json_files[0], "r") as file:
    params = json.load(file)

N = params["N"]
L = np.sqrt(N/params["density"])
# steps = int(params["tau"]/params["dt"])
# number_of_ss = int((steps-start)/params["skip"])
ss = np.arange(0, params["duration"], params["skip"]*params["dt"])
frames = np.argwhere(ss>=start).flatten()
number_of_frames = len(frames)

# Helper functions
def shift_in_mainbox(a):
    return a - L * np.floor((a + L/2) / (L))

def minimum_image_diff(z1, z2):
    
    dx = z1[[0]] - z2[[0]]
    dy = z1[[1]] - z2[[1]]
    # print(dx.shape)
    # if np.abs(dx) > L/2:
    #     dx -= np.sign(dx) * L
    # if np.abs(dy) > L/2:
    #     dy -= np.sign(dy) * L
    dx -= np.where(np.abs(dx) > L / 2, np.sign(dx) * L, 0)
    dy -= np.where(np.abs(dy) > L / 2, np.sign(dy) * L, 0)

    return np.vstack((dx, dy))

def gaussian_kernel(r, sigma=1, r_cut=5):
    k = np.exp(-r**2 / (2 * sigma**2))
    tot_k = np.where(r <= r_cut*sigma, k, 0)
    return tot_k/(2* np.pi * sigma**2)

def bump_kernel(r, r_cut=5):
    k = np.exp(-(r_cut)**2 / (r_cut**2 - r**2))
    tot_k = np.where(r <= r_cut, k, 0)
    return tot_k/(np.pi*r_cut*r_cut*(sp.special.expi(-1)+np.exp(-1)))

print('Loading configurations ...')

with h5py.File(f"{rootdir}snapshots.h5", "r") as f:
    snapshots = np.array(f["configs"]).T[1:]
snapshots = shift_in_mainbox(snapshots[:, :, frames])

# Binning parameters
dx = 1
dy = 1
x_edges = np.arange(-L/2, L/2, dx)
y_edges = np.arange(-L/2, L/2, dy)
x_vals = (x_edges[:-1] + x_edges[1:]) / 2
y_vals = (y_edges[:-1] + y_edges[1:]) / 2
xg, yg = np.meshgrid(x_vals, y_vals, indexing='ij')
n_iters = number_of_frames

# Building density field
density = np.zeros((len(x_vals), len(y_vals), n_iters))
for idx in tqdm(np.ndindex(density.shape[:-1]), total=np.prod(density.shape[:-1])):
    i,j = idx
    coords = np.stack([xg[i, j].ravel(), yg[i, j].ravel()], axis=0)
    r = minimum_image_diff(snapshots[:, :, :n_iters], coords)
    dist = np.linalg.norm(r, axis=0)
    if kernel == "gaussian":
        k = gaussian_kernel(dist, r_cut=rcut)
    elif kernel == "bump":
        k = bump_kernel(dist, r_cut=rcut)
    density[i, j] = np.sum(k, axis=0)

np.save(f"{rootdir}density.npy", density)