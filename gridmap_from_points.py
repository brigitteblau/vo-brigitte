# gridmap_from_points.py (robusto y con guardado en img-code/)
import numpy as np
import argparse
import os
import cv2

def make_grid(points_xy, cell_size=0.05, margin=0.5, max_pixels=20_000_000):
    if len(points_xy) == 0:
        raise ValueError("No hay puntos 2D para grilla.")

    min_xy = points_xy.min(axis=0) - margin
    max_xy = points_xy.max(axis=0) + margin
    size = max_xy - min_xy

    W = int(np.ceil(size[0] / cell_size))
    H = int(np.ceil(size[1] / cell_size))

    # Limitar tamaño de imagen: si es gigante, agrandar celdas automáticamente
    if W * H > max_pixels:
        scale = (W * H / max_pixels) ** 0.5
        cell_size *= scale
        W = int(np.ceil(size[0] / cell_size))
        H = int(np.ceil(size[1] / cell_size))

    grid = np.zeros((H, W), dtype=np.uint8)  # 0 libre, 255 ocupado
    idx = ((points_xy - min_xy) / cell_size).astype(int)
    idx[:, 0] = np.clip(idx[:, 0], 0, W - 1)
    idx[:, 1] = np.clip(idx[:, 1], 0, H - 1)
    grid[idx[:, 1], idx[:, 0]] = 255

    # Relleno suave
    kernel = np.ones((3, 3), np.uint8)
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Visual con "Y arriba"
    grid_vis = np.flipud(grid)
    return grid_vis, (min_xy, max_xy), cell_size

def main():
    p = argparse.ArgumentParser(description="Occupancy grid 2D desde points.npy/ply")
    p.add_argument("--points", type=str, default="points.npy")
    p.add_argument("--cell", type=float, default=0.10, help="tamaño de celda (unidades relativas)")
    p.add_argument("--floor_z_tol", type=float, default=0.05, help="(compat) no usado directamente")
    p.add_argument("--height_thresh", type=float, default=0.15, help="altura mínima sobre piso")
    p.add_argument("--downsample", type=int, default=0, help="si >0, toma 1 de cada N puntos")
    p.add_argument("--save", type=str, default=os.path.join("img-code", "grid.png"))
    p.add_argument("--use_open3d", action="store_true", help="usar Open3D para estimar plano de piso")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)

    # Cargar nube
    if args.points.lower().endswith(".ply"):
        import open3d as o3d
        pc = o3d.io.read_point_cloud(args.points)
        P = np.asarray(pc.points, dtype=np.float32)
    else:
        P = np.load(args.points).astype(np.float32)

    if args.downsample and args.downsample > 1:
        P = P[::args.downsample]

    if P.size == 0:
        raise SystemExit("Nube vacía.")

    # === 1) Estimar plano del piso ===
    if args.use_open3d:
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(P.astype(np.float64))
        plane_model, inliers = pc.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
        a, b, c, d = plane_model
        n = np.array([a, b, c], dtype=np.float64)
        n = n / (np.linalg.norm(n) + 1e-9)
    else:
        n = np.array([0, 0, 1.0], dtype=np.float64)  # aproximación

    # === 2) Alinear normal con +Z === (rotación mínima)
    z_axis = np.array([0, 0, 1.0], dtype=np.float64)
    v = np.cross(n, z_axis)
    s = np.linalg.norm(v)
    c = float(np.dot(n, z_axis))
    if s < 1e-8:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s**2 + 1e-9))
    P_aligned = (R @ P.T).T

    # === 3) Selección de obstáculos según altura ===
    z = P_aligned[:, 2]
    z0 = np.percentile(z, 5)
    mask_obst = z > (z0 + args.height_thresh)

    # Si casi no hay puntos, activar umbral automático (relajar)
    if mask_obst.sum() < 500:
        z95 = np.percentile(z, 95)
        auto_ht = 0.10 * max(1e-6, (z95 - z0))  # 10% del rango útil
        mask_obst = z > (z0 + auto_ht)
        # Si aún quedan pocos, tomar top-30% más altos
        if mask_obst.sum() < 500:
            thr = np.percentile(z, 70)
            mask_obst = z > thr

    XY = P_aligned[mask_obst][:, :2]
    if XY.size == 0:
        # fallback: usa toda la nube (planta)
        XY = P_aligned[:, :2]

    # === 4) Recorte de outliers (98% central)
    lo = np.percentile(XY, 1, axis=0)
    hi = np.percentile(XY, 99, axis=0)
    m = (XY[:, 0] >= lo[0]) & (XY[:, 0] <= hi[0]) & (XY[:, 1] >= lo[1]) & (XY[:, 1] <= hi[1])
    XY = XY[m]

    # === 5) Construir y guardar grid ===
    grid, (mn, mx), cell = make_grid(XY, cell_size=args.cell)
    cv2.imwrite(args.save, grid)

    print(f"[OK] Grid guardado en {args.save}  tamaño={grid.shape[::-1]} celdas  (cell={cell:.3f})")
    print(f"    puntos_obst={len(XY)}  bbox=[{mn[0]:.2f},{mn[1]:.2f}]→[{mx[0]:.2f},{mx[1]:.2f}]")

if __name__ == "__main__":
    main()

