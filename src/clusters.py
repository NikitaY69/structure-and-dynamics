import numpy as np
import json
import h5py
import glob 
import os 
from scipy import sparse
from tqdm import tqdm
from scipy.spatial import KDTree
import pickle
import argparse 

# Parser
parser = argparse.ArgumentParser(
    prog="ABPclust", 
    description=("Compute clusters analysis for ABP simulations.\n"
                 "\n"
                 "Extracts clusters from ABP simulations by analyzing the adjacency matrix of particles.\n" \
                 "It saves in the provided root directory clusters labels for each" \
                 "particle in each snapshot (clusters_labels.npy). \n"
                 "It also returns the radius of gyration for each cluster" \
                 "(rg.pkl)."
    )
)
parser.add_argument(
    "--rootdir", type=str, required=True,
    help="Path to the root directory of the simulation."
)
parser.add_argument(
    "--t0", type=float, default=10,
    help="Wait time (in number of frames) before starting the analysis."
)

# Helper functions
def shift_in_mainbox(a, L):
    return a - L * np.floor((a + L/2) / (L)) + L/2

def build_adjacency(positions, L):
    positions = np.asarray(positions)
    tree = KDTree(positions, boxsize=L)
    touching_radius = 1
    pairs = tree.query_pairs(touching_radius)

    N = len(positions)
    A = np.zeros((N, N), dtype=int)
    for i, j in pairs:
        A[i, j] = 1
        A[j, i] = 1  # Undirected graph

    return A

def unwrap_coords(c, L):
    ref = c[[0]]
    dx = c[:, [0]] - ref[:, [0]]
    dy = c[:, [1]] - ref[:, [1]]
    c[:, [0]] -= np.where(np.abs(dx) > L / 2, np.sign(dx) * L, 0)
    c[:, [1]] -= np.where(np.abs(dy) > L / 2, np.sign(dy) * L, 0)

    return c

def rG(c, L):
    c = unwrap_coords(c, L)
    rCM = np.mean(c, axis=0, keepdims=True)
    d = (c - rCM)
    return np.sqrt(np.mean(np.linalg.norm(d, axis=1)**2))

def minimum_image_diff(z1, z2, L):
    
    dx = z1[:, :, [0]] - z2[:, :, [0]]
    dy = z1[:, :, [1]] - z2[:, :, [1]]
    dx -= np.where(np.abs(dx) > L / 2, np.sign(dx) * L, 0)
    dy -= np.where(np.abs(dy) > L / 2, np.sign(dy) * L, 0)
    r = np.stack(np.stack((dx,dy), axis=-1))
    return r

def main():
    print('Loading configurations ...')
    args = parser.parse_args()

    rootdir = args.rootdir
    start = args.t0

    json_files = glob.glob(os.path.join(rootdir, "*.json"))
    with open(json_files[0], "r") as file:
        params = json.load(file)
    start = round(start/(params["dt"]*params["skip"]))

    N = params["N"]
    L = np.sqrt(N/params["density"])
    # steps = int(params["tau"]/params["dt"])
    # number_of_ss = int((steps-start)/params["skip"])
    ss = np.arange(0, params["duration"], params["skip"]*params["dt"])
    frames = np.argwhere(ss>=start).flatten()
    number_of_frames = len(frames)
    n_iters = number_of_frames

    with h5py.File(f"{rootdir}snapshots.h5", "r") as f:
        s = np.array(f["configs"]).T[1:]

    data = np.genfromtxt(f"{rootdir}obs.txt", delimiter='', names=True)

    if type(data["t"]) is not np.ndarray or len(data["t"]) != len(ss):
        print("Crashed simulation: not enough frames.")
        exit()

    snapshots = shift_in_mainbox(s[:, :, frames], L).T

    clusters_labels = np.empty((n_iters, N), dtype=int)
    clusters = []
    clustered_ss = []
    rGS = []
    # inter_dists = []

    for i in tqdm(range(n_iters)):
        a = build_adjacency(snapshots[i], L)
        n_components, labels = sparse.csgraph.connected_components(a, directed=False)
        clusters_labels[i] = labels
        unique_labels = np.unique(labels)
        clusters.append([np.where(labels == label)[0] for label in unique_labels])
        clustered_ss.append([snapshots[i][c] for c in clusters[i]])
        rGS.append(np.array([rG(c, L) for c in clustered_ss[i]]))
        # inter_dists.append(np.concatenate([np.linalg.norm(minimum_image_diff(c[:, None, :], c[None, :, :]), axis=3).flatten() for c in clustered_ss[i]]))

    np.save(f'{rootdir}clusters_labels.npy', clusters_labels)
    # with open(f'{rootdir}clusters.pkl', 'wb') as f:
    #     pickle.dump(clusters, f)

    with open(f'{rootdir}rg.pkl', 'wb') as f:
        pickle.dump(rGS, f)

    # with open(f'{rootdir}pairprob.pkl', 'wb') as f:
    #     pickle.dump(inter_dists, f)