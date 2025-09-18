
#estoy graficando la trayectoria de trajectory.npy
#un archivo .npy es para guardar matrices (tipo y la forma)
# plot_traj.py
import argparse, numpy as np, os, matplotlib.pyplot as plt

def plot_traj_cli(traj_path, out_img):
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    T = np.load(traj_path)  # (N,3)
    plt.figure()
    plt.plot(T[:,0], T[:,2])  # x vs z (vista planta)
    plt.xlabel("X"); plt.ylabel("Z"); plt.title("Trayectoria (planta)")
    plt.axis("equal"); plt.grid(True)
    plt.savefig(out_img, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"🖼 guardado {out_img}")
    return out_img

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plot_traj_cli(a.traj, a.out)
