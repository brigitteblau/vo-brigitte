import numpy as np
import matplotlib.pyplot as plt
#estoy graficando la trayectoria de trajectory.npy
#un archivo .npy es para guardar matrices (tipo y la forma)
def main():
    traj = np.load("trajectory.npy")
    print("[INFO] Trayectoria cargada, forma:", traj.shape)

    plt.figure()
    plt.plot(traj[:,0], traj[:,1], "-o", markersize=3)
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Trayectoria monocular (escala relativa)")
    plt.show()

if __name__ == "__main__":
    main()
