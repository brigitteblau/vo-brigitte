# vo_triangulate.py  (compat con main.py)
import cv2, numpy as np, argparse, os, sys

def log(*a):
    print(*a); sys.stdout.flush()

def open_source(input_src):
    if str(input_src).isdigit():
        cap = cv2.VideoCapture(int(input_src), cv2.CAP_DSHOW)  # mejor en Windows
    else:
        if not os.path.exists(input_src):
            raise FileNotFoundError(f"No existe: {input_src}")
        cap = cv2.VideoCapture(input_src)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir: {input_src}")
    return cap

def guess_K(shape):
    h, w = shape[:2]
    focal = 0.9 * max(w, h)
    return np.array([[focal, 0, w/2],
                     [0, focal, h/2],
                     [0,     0,   1]], dtype=np.float64)

def orb_knn(img1, img2, orb, ratio):
    kp1, d1 = orb.detectAndCompute(img1, None)
    kp2, d2 = orb.detectAndCompute(img2, None)
    if d1 is None or d2 is None or len(d1) == 0 or len(d2) == 0:
        return [], kp1, kp2
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(d1, d2, k=2)
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good, kp1, kp2

def pts_from_matches(kp1, kp2, ms):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in ms])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in ms])
    return pts1, pts2

def compose(R12, t12, Rcw1, tcw1):
    Rcw2 = R12 @ Rcw1
    tcw2 = R12 @ tcw1 + t12
    return Rcw2, tcw2

def cam_center(Rcw, tcw):
    # C = -R^T t
    return (-Rcw.T @ tcw).ravel()

def triangulate_two(K, Rcw_a, tcw_a, Rcw_b, tcw_b, uv_a, uv_b):
    P1 = K @ np.hstack([Rcw_a, tcw_a])
    P2 = K @ np.hstack([Rcw_b, tcw_b])
    X4 = cv2.triangulatePoints(P1, P2, uv_a.T, uv_b.T)
    X3 = (X4[:3] / (X4[3] + 1e-9)).T
    return X3

def reproj_err(K, Rcw, tcw, Xw, uv):
    Xc = (Rcw @ Xw.T + tcw).T
    z = Xc[:, 2:3]
    proj = (K @ (Xc.T)).T
    proj = proj[:, :2] / np.maximum(z, 1e-9)
    err = np.linalg.norm(proj - uv, axis=1)
    return err, Xc

def to_ply(points_xyz, path="points.ply", colors=None):
    pts = points_xyz.astype(np.float32)
    with open(path, "w") as f:
        if colors is None:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for p in pts: f.write(f"{p[0]} {p[1]} {p[2]}\n")
        else:
            cols = colors.astype(np.uint8)
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for p, c in zip(pts, cols): f.write(f"{p[0]} {p[1]} {p[2]} {int(c[2])} {int(c[1])} {int(c[0])}\n")
    log(f"💾 guardado {path} ({len(pts)} puntos)")
    return path

def triangulate_cli(input_src, out_ply="points.ply", max_frames=None,
                    max_feats=3500, ratio=0.75, ransac_th=2.0, reproj_th=4.0,
                    kf_stride=2, show=False, save_colors=False):
    cap = open_source(input_src)

    ok, f0 = cap.read()
    if not ok: raise RuntimeError("No pude leer el primer frame.")
    gray_prev = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    K = guess_K(f0.shape)
    log("[INFO] K:\n", K)

    orb = cv2.ORB_create(max_feats)

    Rcw_prev = np.eye(3); tcw_prev = np.zeros((3,1))
    traj = [cam_center(Rcw_prev, tcw_prev)]

    # init KF
    kf_gray = gray_prev.copy()
    Rcw_kf, tcw_kf = Rcw_prev.copy(), tcw_prev.copy()
    kf_idx = 0

    all_points = []
    all_colors = []

    traj_canvas = np.ones((500,500,3), np.uint8)*255
    frame_idx = 0
    total_kf = 0
    total_triang = 0

    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            log("[INFO] Fin de video."); break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_idx += 1

        matches, kp1, kp2 = orb_knn(gray_prev, gray, orb, ratio)
        log(f"[DBG] Frame {frame_idx}: matches={len(matches)}")
        if len(matches) >= 8:
            pts1, pts2 = pts_from_matches(kp1, kp2, matches)
            E, maskE = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=ransac_th)
            if E is not None and maskE is not None and int(maskE.sum()) >= 8:
                _, R, t, maskP = cv2.recoverPose(E, pts1, pts2, K)
                Rcw_curr, tcw_curr = compose(R, t, Rcw_prev, tcw_prev)
                traj.append(cam_center(Rcw_curr, tcw_curr))

                # KF por stride fijo
                if frame_idx % kf_stride == 0:
                    total_kf += 1
                    log(f"[KF] usando KF #{total_kf} (prev idx {kf_idx}) vs frame {frame_idx}")
                    m2, kp_kf, kp_c = orb_knn(kf_gray, gray, orb, ratio)
                    log(f"[DBG]   matches KF={len(m2)}")
                    if len(m2) >= 20:
                        uv_kf, uv_c = pts_from_matches(kp_kf, kp_c, m2)
                        X = triangulate_two(K, Rcw_kf, tcw_kf, Rcw_curr, tcw_curr, uv_kf, uv_c)
                        err1, Xc1 = reproj_err(K, Rcw_kf,  tcw_kf,  X, uv_kf)
                        err2, Xc2 = reproj_err(K, Rcw_curr, tcw_curr, X, uv_c)
                        valid = (Xc1[:,2] > 0) & (Xc2[:,2] > 0) & (err1 < reproj_th) & (err2 < reproj_th)
                        good = X[valid]
                        log(f"[KF]   triangulados validos={len(good)}")
                        if len(good) > 0:
                            all_points.append(good); total_triang += len(good)
                            if save_colors:
                                v = uv_c[valid].astype(int)
                                h, w = gray.shape
                                v[:,0] = np.clip(v[:,0], 0, w-1); v[:,1] = np.clip(v[:,1], 0, h-1)
                                cols = frame[v[:,1], v[:,0], :] if frame.ndim==3 else np.full((len(v),3), 200)
                                all_colors.append(cols.astype(np.uint8))
                    # actualizar KF al actual
                    kf_gray = gray.copy(); Rcw_kf, tcw_kf = Rcw_curr.copy(), tcw_curr.copy(); kf_idx = frame_idx

                if show:
                    traj_canvas[:] = 255
                    pts3 = np.array(traj, np.float32)  # (N,3)
                    if len(pts3) >= 2:
                        pts2d = pts3[:, [0, 2]]  # planta XZ
                        mn = pts2d.min(0); mx = pts2d.max(0); span = np.maximum(mx - mn, 1e-3)
                        m = 30; W, H = traj_canvas.shape[1], traj_canvas.shape[0]
                        s = 0.9 * min((W-2*m)/span[0], (H-2*m)/span[1])
                        p = (pts2d - mn) * s + np.array([m, m])
                        for i in range(1, len(p)):
                            cv2.line(traj_canvas, tuple(p[i-1].astype(int)), tuple(p[i].astype(int)), (0,0,0), 2)
                        cv2.circle(traj_canvas, tuple(p[-1].astype(int)), 4, (0,0,255), -1)
                    cv2.imshow("Trayectoria 2D", traj_canvas)

                    inl = maskP.ravel().astype(bool)
                    inliers = [mm for (mm, ok) in zip(matches, inl) if ok]
                    vis = cv2.drawMatches(cv2.cvtColor(gray_prev, cv2.COLOR_GRAY2BGR), kp1,
                                          cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), kp2,
                                          inliers, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    cv2.putText(vis, f"Inliers:{int(maskP.sum())}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    cv2.imshow("Matches (inliers)", vis)

                Rcw_prev, tcw_prev = Rcw_curr, tcw_curr

        gray_prev = gray
        if show and (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release(); cv2.destroyAllWindows()

    # Guardados
    traj_arr = np.array(traj, dtype=np.float32)     # (N,3) xyz
    np.save("trajectory.npy", traj_arr); log(f"[OK] trajectory.npy N={len(traj_arr)}")

    P = np.vstack(all_points) if len(all_points)>0 else np.zeros((0,3), np.float32)
    np.save("points.npy", P); log(f"[OK] points.npy N={len(P)}")

    C = None
    if save_colors:
        C = np.vstack(all_colors) if len(all_colors)>0 else np.zeros((0,3), np.uint8)
        np.save("colors.npy", C); log(f"[OK] colors.npy N={len(C)}")

    ply_path = to_ply(P, path="points.ply", colors=C)
    log(f"[RESUMEN] keyframes={int(np.ceil(frame_idx/max(1,kf_stride)))}  puntos_total={len(P)}")
    return ply_path

def parse_args():
    p = argparse.ArgumentParser(description="VO + Triangulación (KF + filtros)")
    p.add_argument("--input", required=False, default="test1.mp4", help="Ruta de video o índice de cámara (0,1)")
    p.add_argument("--out",   required=False, default="points.ply")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--max_feats",  type=int, default=3500)
    p.add_argument("--ratio",      type=float, default=0.75)
    p.add_argument("--ransac_th",  type=float, default=2.0)
    p.add_argument("--reproj_th",  type=float, default=4.0)
    p.add_argument("--kf_stride",  type=int, default=2)
    p.add_argument("--show", action="store_true")
    p.add_argument("--save_colors", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    a = parse_args()
    triangulate_cli(a.input, a.out, a.max_frames, a.max_feats, a.ratio, a.ransac_th,
                    a.reproj_th, a.kf_stride, a.show, a.save_colors)
