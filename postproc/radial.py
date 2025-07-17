import numpy as np
import json
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
plt.style.use('physrev')
from matplotlib.ticker import AutoMinorLocator, LogLocator

import argparse 
from tqdm import tqdm
import os 
import time
import h5py
import glob

# Parser
parser = argparse.ArgumentParser(
    prog="ABPrdf", 
    description=("Compute ABP radial distribution function."
    )
)
parser.add_argument(
    "--rootdir", type=str, required=True,
    help="Path to the root directory of the simulation."
)
parser.add_argument(
    "--t0", type=float, default=0,
    help="Wait time (in time units) before starting the analysis."
)
parser.add_argument(
    "--skip", type=float, default=1,
    help="Skip (in time units) between snapshots."
)
parser.add_argument(
    "--sweep", type=float, default=0.05,
    help="Bins width."
)
# parser.add_argument(
#     "--max", type=float, default=5,
#     help="Maximum distance to probe pair correlations."
# )
parser.add_argument("--remake", action="store_true", help="Remake calculations even if already saved.")
args = parser.parse_args()

t0 = time.time()
# Loading parameters
rootdir = args.rootdir
start = args.t0
remake = args.remake
sweep = args.sweep
# max_r = args.max

json_files = glob.glob(os.path.join(rootdir, "*.json"))
with open(json_files[0], "r") as file:
    params = json.load(file)
N = params["N"]
L = np.sqrt(N/params["density"])
# steps = int(params["tau"]/params["dt"])
# start = params["t0"]

skip = round(args.skip/(params["dt"]*params["skip"]))
# number_of_ss = int((steps-start)/params["skip"])
ss = np.arange(0, params["duration"], params["skip"]*params["dt"])
frames = np.argwhere(ss>=start).flatten()[::skip]
number_of_frames = len(frames)
print(number_of_frames)

# Helper functions
def shift_in_mainbox(a):
    return a - L * np.floor((a + L/2) / (L))

def minimum_image_dist(a, b):
    return L/2 - np.abs(np.abs(a-b)-L/2)

def minimum_image_diff(z1, z2):
    dx = z1.real - z2.real
    dy = z1.imag - z2.imag
    # print(dx.shape)
    # if np.abs(dx) > L/2:
    #     dx -= np.sign(dx) * L
    # if np.abs(dy) > L/2:
    #     dy -= np.sign(dy) * L
    dx -= np.where(np.abs(dx) > L / 2, np.sign(dx) * L, 0)
    dy -= np.where(np.abs(dy) > L / 2, np.sign(dy) * L, 0)

    return dx + 1j * dy

def principal_angle(A):
    return (A + np.pi) % (2 * np.pi) - np.pi

# jPDF parameters
max_r = L/2
r_edges = np.arange(0, max_r, sweep)
r_values = (r_edges[:-1] + r_edges[1:]) / 2
if f"radial.npy" not in os.listdir(rootdir) or remake:
    # Loading configurations
    positions = np.empty(shape=(number_of_frames, N), dtype=complex)
    print('Loading configurations ...')
    for i, frame in tqdm(enumerate(frames), total=number_of_frames):
        with h5py.File(f"{rootdir}snapshots.h5", "r") as f:
            data = f["configs"][frame].T
        pos = shift_in_mainbox(data[1:])
        positions[i] = pos[0] + 1j * pos[1]

    corr = []
    print('Computing correlations ...')
    for i in tqdm(range(N)):
        # Translating origin
        centered_pos = minimum_image_diff(positions, positions[:, [i]])
        
        # Computing the pair-correlation
        centered_pos = np.delete(centered_pos, i, axis=1)
        radius = np.abs(centered_pos).flatten()
        corr.append(np.histogram(radius, bins=r_edges)[0]/number_of_frames)

    corr = np.sum(np.array(corr), axis=0)/(2*np.pi*r_values*sweep*params["density"]*N)
    np.save(f"{rootdir}radial.npy", corr)

else:
    print('Correlations already computed. Loading...')
    corr = np.load(f"{rootdir}radial.npy")

# Plot
fig, ax = plt.subplots()
plt.plot(r_values, corr, ms=4)
plt.xlabel(r"$r$")
plt.ylabel(r"$g(r)$")
plt.xlim(0, max_r)

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

plt.savefig(f"{rootdir}radial.jpg", bbox_inches='tight', dpi=400)
elapsed_time = (time.time() - t0)/60 # minutes
print(f"Execution time: {elapsed_time:.3f} minutes")

if __name__ == "__main__":
    main()