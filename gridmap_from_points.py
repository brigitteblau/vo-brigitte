#gridmap_from_points.py
import numpy as np
import argparse
import cv2

def make_grid(points_xy, cell_size=0.05, margin=0.5):
    """
    points_xy: Nx2 (metros en escala relativa)
    cell_size: tamaño de celda (ej. 5 cm)
    margin: borde extra alrededor del bounding box (en mismas unidades que los puntos)
    """
    if len(points_xy) == 0:
        raise ValueError("No hay puntos 2D para grilla.")

    min_xy = points_xy.min(axis=0) - margin
    max_xy = points_xy.max(axis=0) + margin
    size = max_xy - min_xy
    W = int(np.ceil(size[0] / cell_size))
    H = int(np.ceil(size[1] / cell_size))

    grid = np.zeros((H, W), dtype=np.uint8)  # 0 = libre/desconocido, 255 = ocupado

    # marcar ocupación donde caen puntos
    idx = ((points_xy - min_xy) / cell_size).astype(int)
    idx[:,0] = np.clip(idx[:,0], 0, W-1)
    idx[:,1] = np.clip(idx[:,1], 0, H-1)
    grid[idx[:,1], idx[:,0]] = 255

    # Suavizado morfológico para rellenar agujeros pequeños
    kernel = np.ones((3,3), np.uint8)
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Voltear vertical para visualizar “Y arriba”
    grid_vis = np.flipud(grid)

    return grid_vis, (min_xy, max_xy), cell_size

def main():
    p = argparse.ArgumentParser(description="Occupancy grid 2D desde points.npy")
    p.add_argument("--points", type=str, default="points.npy")
    p.add_argument("--cell", type=float, default=0.05, help="tamaño de celda (m relativos)")
    p.add_argument("--floor_z_tol", type=float, default=0.05, help="tolerancia en Z para piso tras alineado (m)")
    p.add_argument("--height_thresh", type=float, default=0.15, help="altura mínima (sobre piso) para considerar obstáculo (m)")
    p.add_argument("--downsample", type=int, default=0, help="si >0, toma 1 de cada N puntos")
    p.add_argument("--save", type=str, default="grid.png", help="salida PNG")
    p.add_argument("--use_open3d", action="store_true", help="usar Open3D para detectar plano de piso")
    args = p.parse_args()

    P = np.load(args.points)  # Nx3
    if args.downsample and args.downsample > 1:
        P = P[::args.downsample]

    if len(P) == 0:
        raise SystemExit("points.npy está vacío.")

    # === 1) Estimar plano de piso ===
    # Opción Open3D (robusto y simple)
    if args.use_open3d:
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(P.astype(np.float64))
        plane_model, inliers = pc.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
        a,b,c,d = plane_model
        n = np.array([a,b,c], dtype=np.float64)
        n = n / (np.linalg.norm(n) + 1e-9)
    else:
        # Fallback muy simple: usar la mediana de Z como aproximación de piso
        n = np.array([0,0,1.0], dtype=np.float64)

    # === 2) Alinear normal del piso con +Z ===
    z_axis = np.array([0,0,1.0], dtype=np.float64)
    v = np.cross(n, z_axis)
    s = np.linalg.norm(v)
    c = float(np.dot(n, z_axis))
    if s < 1e-8:
        R = np.eye(3)
    else:
        vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s**2 + 1e-9))

    P_aligned = (R @ P.T).T
    z = P_aligned[:,2]

    # Estimar nivel de piso (z0) como percentil bajo
    z0 = np.percentile(z, 5)
    # Puntos considerados obstáculos: por encima de z0 + height_thresh
    mask_obst = z > (z0 + args.height_thresh)
    XY = P_aligned[mask_obst][:, :2]

    grid, (mn, mx), cell = make_grid(XY, cell_size=args.cell)
    cv2.imwrite(args.save, grid)
    print(f"[OK] Grid guardado en {args.save}  tamaño={grid.shape[::-1]} celdas  (cell={cell} m)")
    print("Tip: abrilo con cualquier visor de imágenes. Blanco = ocupado.")

if __name__ == "__main__":
    main()
