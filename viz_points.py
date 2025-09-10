import numpy as np

def show_matplotlib(points, colors=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = points[:,0]; Y = points[:,1]; Z = points[:,2]
    if colors is not None:
        # colors esperados en RGB [0..255]
        C = colors / 255.0
        ax.scatter(X, Y, Z, s=1, c=C)
    else:
        ax.scatter(X, Y, Z, s=1)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_box_aspect([1,1,1])
    plt.title("Sparse point cloud")
    plt.show()

def show_open3d(points, colors=None):
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        # Open3D espera float [0..1] en RGB
        C = colors[:, ::-1] / 255.0  # si guardaste BGR -> RGB
        pc.colors = o3d.utility.Vector3dVector(C)
    o3d.visualization.draw_geometries([pc])

if __name__ == "__main__":
    P = np.load("points.npy")  # Nx3
    try:
        C = np.load("colors.npy")
    except Exception:
        C = None

    try:
        import open3d  # noqa: F401
        show_open3d(P, C)
    except Exception:
        print("[INFO] Open3D no disponible, usando Matplotlib 3D.")
        show_matplotlib(P, C)
