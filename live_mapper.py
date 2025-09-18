# -*- coding: utf-8 -*-
"""
live_mapper.py — VO (camera/video/images) + triangulación + grid + trayectoria
Uso:
  # Cámara (índice 0)
  python live_mapper.py --camera 0

  # Video
  python live_mapper.py --video data/recorrido.mp4

  # Carpeta de imágenes (orden alfanumérico)
  python live_mapper.py --images_dir data/frames --ext .png

Flags útiles:
  --cell 0.15 --grid_m 30 --kf_stride 3 --save_every 30 --no_show
  --draw_on_last      (dibuja trayectoria sobre el último frame y lo guarda)
  --poses_csv poses.csv
"""
import os, glob, argparse, time
import numpy as np
import cv2

# -------------------- helpers esperados de tu módulo --------------------
try:
    from vo_triangulate import (
        open_source, guess_K, orb_knn, pts_from_matches, compose,
        triangulate_two, reproj_err
    )
except Exception:
    # Fallbacks mínimos si no tienes el módulo a mano (no triangulan).
    def open_source(src: str):
        return cv2.VideoCapture(0 if src in [None, "0", "1", "2"] else src)

    def guess_K(shape_bgr):
        """ Aproxima K suponiendo fx ≈ fy ≈ 0.9 * max(W,H) y cx,cy = centro """
        if hasattr(shape_bgr, "shape"):
            H, W = shape_bgr.shape[:2]
        else:
            H, W = shape_bgr
        f = 0.9 * max(H, W)
        K = np.array([[f, 0, W/2],
                      [0, f, H/2],
                      [0, 0, 1]], dtype=np.float64)
        return K

    def orb_knn(img1, img2, orb, ratio=0.75):
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        if des1 is None or des2 is None: return [], [], []
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = bf.knnMatch(des1, des2, k=2)
        good = []
        for m,n in knn:
            if m.distance < ratio * n.distance:
                good.append(m)
        return good, kp1, kp2

    def pts_from_matches(kp1, kp2, matches):
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        return pts1, pts2

    def compose(R, t, Rcw_prev, tcw_prev):
        # Pose compuesta: (Rcw_cur, tcw_cur) = (R * Rcw_prev, R*tcw_prev + t)
        Rcw_cur = R @ Rcw_prev
        tcw_cur = R @ tcw_prev + t
        return Rcw_cur, tcw_cur

# -------------------- Ocupancy Grid --------------------
class OccGrid:
    def __init__(self, cell=0.15, grid_m=30.0):
        self.cell = float(cell)
        n = max(8, int(np.ceil(grid_m / self.cell)))
        self.W = self.H = n
        self.grid = np.zeros((self.H, self.W), np.uint8)
        self.origin = np.array([-self.W/2 * self.cell, -self.H/2 * self.cell], dtype=np.float64)

    def world_to_grid(self, XY):
        if XY.size == 0: return np.empty((0,2), dtype=int)
        ij = np.floor((XY - self.origin) / self.cell).astype(int)
        m = (ij[:,0]>=0)&(ij[:,0]<self.W)&(ij[:,1]>=0)&(ij[:,1]<self.H)
        return ij[m]

    def update(self, XY):
        ij = self.world_to_grid(XY)
        if ij.size: self.grid[ij[:,1], ij[:,0]] = 255

    def recenter_if_needed(self, cam_xy, margin_frac=0.2):
        cam_ij = np.floor((cam_xy - self.origin) / self.cell).astype(int)
        low = int(self.W * margin_frac); high = int(self.W * (1 - margin_frac))
        shift_i = shift_j = 0
        if cam_ij[0] < low:   shift_i = cam_ij[0] - low
        if cam_ij[0] > high:  shift_i = cam_ij[0] - high
        if cam_ij[1] < low:   shift_j = cam_ij[1] - low
        if cam_ij[1] > high:  shift_j = cam_ij[1] - high
        if shift_i or shift_j:
            self.grid = np.roll(self.grid, -shift_j, axis=0)
            self.grid = np.roll(self.grid, -shift_i, axis=1)
            # limpiar bordes introducidos por el roll
            if shift_j != 0:
                if shift_j > 0:   self.grid[-shift_j:, :] = 0
                else:             self.grid[: -shift_j, :] = 0
            if shift_i != 0:
                if shift_i > 0:   self.grid[:, -shift_i:] = 0
                else:             self.grid[:, : -shift_i] = 0
            self.origin += np.array([shift_i*self.cell, shift_j*self.cell], dtype=np.float64)

    def vis(self, scale=3):
        ker = np.ones((3,3), np.uint8)
        g = cv2.morphologyEx(self.grid, cv2.MORPH_CLOSE, ker, iterations=1)
        g = np.flipud(g)  # Y hacia arriba
        return cv2.resize(g, (self.W*scale, self.H*scale), interpolation=cv2.INTER_NEAREST)

# -------------------- Lectores de frames --------------------
def frames_from_camera(index):
    cap = cv2.VideoCapture(int(index))
    if not cap.isOpened(): raise RuntimeError(f"No pude abrir cámara {index}")
    while True:
        ok, frame = cap.read()
        if not ok: break
        yield frame
    cap.release()

def frames_from_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): raise RuntimeError(f"No pude abrir video: {path}")
    while True:
        ok, frame = cap.read()
        if not ok: break
        yield frame
    cap.release()

def frames_from_folder(folder, ext=".png", sort_numeric=True):
    paths = sorted(glob.glob(os.path.join(folder, f"*{ext}")))
    if sort_numeric:
        def key(p):
            b = os.path.basename(p)
            d = "".join([c for c in b if c.isdigit()])
            return int(d) if d.isdigit() else b
        paths = sorted(paths, key=key)
    if not paths: raise RuntimeError(f"No encontré imágenes en {folder} con ext {ext}")
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None: continue
        yield img

# -------------------- Main VO + (opcional) triangulación y grid --------------------
def run_mapper(source_type, source_value,
               cell=0.15, grid_m=30.0, kf_stride=3,
               max_feats=1500, ratio=0.75, ransac_th=2.0, reproj_th=4.0,
               show=True, save_every=30,
               draw_on_last=False, poses_csv="poses.csv"):
    os.makedirs("img-code", exist_ok=True)

    # Generador de frames según fuente
    if source_type == "camera":
        gen = frames_from_camera(source_value)
    elif source_type == "video":
        gen = frames_from_video(source_value)
    elif source_type == "images":
        folder, ext = source_value
        gen = frames_from_folder(folder, ext)
    else:
        raise ValueError("source_type inválido")

    # Primer frame
    try:
        first = next(gen)
    except StopIteration:
        raise RuntimeError("Fuente sin frames.")
    gray_prev = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    K = guess_K(first.shape)
    orb = cv2.ORB_create(max_feats)

    # Pose world (Rcw, tcw)
    Rcw_prev, tcw_prev = np.eye(3), np.zeros((3,1))
    kf_gray, Rcw_kf, tcw_kf = gray_prev.copy(), Rcw_prev.copy(), tcw_prev.copy()

    og = OccGrid(cell=cell, grid_m=grid_m)
    traj_img = np.zeros((600, 600, 3), np.uint8)  # lienzo de trayectoria
    scale_traj = 40.0  # factor de dibujo (ajusta a gusto)
    center = (traj_img.shape[1]//2, traj_img.shape[0]//2)

    frame_idx, saved = 0, 0
    last_frame_vis = first.copy()
    poses = [(0.0, 0.0, 0.0)]  # x,z,yaw aproximado (yaw ~ t relativo pequeño)

    for frame in [first] + list(gen):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_idx += 1

        matches, kp1, kp2 = orb_knn(gray_prev, gray, orb, ratio)
        if len(matches) >= 8:
            pts1, pts2 = pts_from_matches(kp1, kp2, matches)
            E, maskE = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=ransac_th)
            if E is not None and maskE is not None and int(maskE.sum()) >= 8:
                _, R, t, maskP = cv2.recoverPose(E, pts1, pts2, K)
                Rcw_cur, tcw_cur = compose(R, t, Rcw_prev, tcw_prev)

                # Keyframe y triangulación
                if kf_stride > 0 and (frame_idx % kf_stride == 0):
                    m2, kp_kf, kp_c = orb_knn(kf_gray, gray, orb, ratio)
                    if len(m2) >= 20 and 'triangulate_two' in globals():
                        uv_kf, uv_c = pts_from_matches(kp_kf, kp_c, m2)
                        try:
                            X = triangulate_two(K, Rcw_kf, tcw_kf, Rcw_cur, tcw_cur, uv_kf, uv_c)
                            err1, Xc1 = reproj_err(K, Rcw_kf,  tcw_kf,  X, uv_kf)
                            err2, Xc2 = reproj_err(K, Rcw_cur, tcw_cur, X, uv_c)
                            valid = (Xc1[:,2] > 0) & (Xc2[:,2] > 0) & (err1 < reproj_th) & (err2 < reproj_th)
                            P = X[valid]
                            XY = P[:, [0, 2]]  # planta XZ
                            og.update(XY)
                        except Exception:
                            pass

                        cam_xyz = (-Rcw_cur.T @ tcw_cur).ravel()
                        og.recenter_if_needed(np.array([cam_xyz[0], cam_xyz[2]]))

                        kf_gray, Rcw_kf, tcw_kf = gray.copy(), Rcw_cur.copy(), tcw_cur.copy()

                # Dibujo de trayectoria aproximando traslación en X,Z
                cam_xyz = (-Rcw_cur.T @ tcw_cur).ravel()
                x, z = float(cam_xyz[0]), float(cam_xyz[2])
                poses.append((x, z, 0.0))
                x_pix = int(center[0] + x * scale_traj)
                z_pix = int(center[1] - z * scale_traj)
                cv2.circle(traj_img, (x_pix, z_pix), 2, (0,255,0), -1)

                Rcw_prev, tcw_prev = Rcw_cur, tcw_cur

        gray_prev = gray
        last_frame_vis = frame.copy()

        # Visualización
        if show:
            grid_vis = og.vis(scale=3)
            cv2.imshow("Grid (ESC para salir)", grid_vis)
            cv2.imshow("Trayectoria", traj_img)

        # Guardados periódicos
        if save_every > 0 and frame_idx % save_every == 0:
            cv2.imwrite(os.path.join("img-code", f"grid_live_{saved:05d}.png"), og.vis(scale=2))
            cv2.imwrite(os.path.join("img-code", f"traj_{saved:05d}.png"), traj_img)
            saved += 1

        if show:
            if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                break

    # Guardados finales
    final_grid = os.path.join("img-code", "grid_live_final.png")
    cv2.imwrite(final_grid, og.vis(scale=2))
    final_traj = os.path.join("img-code", "traj_final.png")
    cv2.imwrite(final_traj, traj_img)

    # (Opcional) Dibujo de la trayectoria sobre el último frame
    if draw_on_last:
        overlay = last_frame_vis.copy()
        # Proyectamos nuestros puntos de trayectoria en una mini-ventana arriba a la derecha
        h, w = overlay.shape[:2]
        small = cv2.resize(traj_img, (min(300, w//3), min(300, h//3)))
        oh, ow = small.shape[:2]
        overlay[10:10+oh, w-10-ow:w-10] = small
        last_out = os.path.join("img-code", "last_with_traj.png")
        cv2.imwrite(last_out, overlay)

    # Export poses
    if poses_csv:
        import csv
        with open(poses_csv, "w", newline="") as f:
            wri = csv.writer(f)
            wri.writerow(["x", "z", "yaw_rad"])
            wri.writerows(poses)

    print("Listo:")
    print("  -", final_grid)
    print("  -", final_traj)
    if draw_on_last:
        print("  - img-code/last_with_traj.png")
    if poses_csv:
        print("  -", poses_csv)

# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser("Live / Batch VO Mapper")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--camera", type=int, help="índice de cámara (0,1,...)")
    g.add_argument("--video", type=str, help="ruta a video")
    g.add_argument("--images_dir", type=str, help="carpeta con imágenes secuenciales")

    ap.add_argument("--ext", type=str, default=".png", help="extensión de imágenes si usas --images_dir")
    ap.add_argument("--cell", type=float, default=0.15)
    ap.add_argument("--grid_m", type=float, default=30.0, help="ancho/alto del mapa en unidades VO")
    ap.add_argument("--kf_stride", type=int, default=3, help="cada cuántos frames triangulamos/actualizamos grid")
    ap.add_argument("--max_feats", type=int, default=1500)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--ransac_th", type=float, default=2.0)
    ap.add_argument("--reproj_th", type=float, default=4.0)
    ap.add_argument("--no_show", action="store_true")
    ap.add_argument("--save_every", type=int, default=30, help="guardar snapshot cada N frames (0=off)")
    ap.add_argument("--draw_on_last", action="store_true", help="pegar mini trayectoria al último frame")
    ap.add_argument("--poses_csv", type=str, default="poses.csv")
    return ap.parse_args()

if __name__ == "__main__":
    a = parse_args()
    if a.camera is not None:
        src_type, src_val = "camera", a.camera
    elif a.video:
        src_type, src_val = "video", a.video
    else:
        src_type, src_val = "images", (a.images_dir, a.ext)

    run_mapper(src_type, src_val,
               cell=a.cell, grid_m=a.grid_m, kf_stride=a.kf_stride,
               max_feats=a.max_feats, ratio=a.ratio, ransac_th=a.ransac_th, reproj_th=a.reproj_th,
               show=not a.no_show, save_every=a.save_every,
               draw_on_last=a.draw_on_last, poses_csv=a.poses_csv)
