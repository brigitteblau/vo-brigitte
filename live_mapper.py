# rehacer este archivo. abrir camara marcar trayecto a poarir de fotos
#ddespues una ultima foto 
#quizas sea mejor un video??
#live_mapper.py — VO + triangulación + grid en TIEMPO REAL
import os, time, argparse, numpy as np, cv2
from vo_triangulate import open_source, guess_K, orb_knn, pts_from_matches, compose, triangulate_two, reproj_err

class OccGrid:
    def __init__(self, cell=0.15, grid_m=30.0):
        """
        cell: tamaño de celda (unidades VO, p.ej. ~0.1)
        grid_m: ancho/alto del mapa en mismas unidades (cuadrado)
        """
        self.cell = float(cell)
        n = max(8, int(np.ceil(grid_m / self.cell)))
        self.W = self.H = n
        self.grid = np.zeros((self.H, self.W), np.uint8)
        # origen (min corner) y centro
        self.origin = np.array([-self.W/2 * self.cell, -self.H/2 * self.cell], dtype=np.float64)

    def world_to_grid(self, XY):
        ij = np.floor((XY - self.origin) / self.cell).astype(int)
        mask = (ij[:,0] >= 0) & (ij[:,0] < self.W) & (ij[:,1] >= 0) & (ij[:,1] < self.H)
        return ij[mask]

    def update(self, XY):
        if XY.size == 0: return
        ij = self.world_to_grid(XY)
        if ij.size == 0: return
        self.grid[ij[:,1], ij[:,0]] = 255

    def recenter_if_needed(self, cam_xy, margin_frac=0.2):
        """Si la cámara se acerca al borde, mover el mapa (rolling) para mantenerla en el centro."""
        cam_ij = np.floor((cam_xy - self.origin) / self.cell).astype(int)
        low = int(self.W * margin_frac); high = int(self.W * (1 - margin_frac))
        shift_i = shift_j = 0
        if cam_ij[0] < low:   shift_i = cam_ij[0] - low
        if cam_ij[0] > high:  shift_i = cam_ij[0] - high
        if cam_ij[1] < low:   shift_j = cam_ij[1] - low
        if cam_ij[1] > high:  shift_j = cam_ij[1] - high
        if shift_i != 0 or shift_j != 0:
            self.grid = np.roll(self.grid, -shift_j, axis=0)
            self.grid = np.roll(self.grid, -shift_i, axis=1)
            # limpiar bordes introducidos por el roll
            if shift_j != 0:
                rows = slice(self.H + (-shift_j) if shift_j<0 else 0,
                             self.H if shift_j>0 else -shift_j)
                self.grid[rows, :] = 0
            if shift_i != 0:
                cols = slice(self.W + (-shift_i) if shift_i<0 else 0,
                             self.W if shift_i>0 else -shift_i)
                self.grid[:, cols] = 0
            # actualizar origen en mundo
            self.origin += np.array([shift_i*self.cell, shift_j*self.cell], dtype=np.float64)

    def vis(self, scale=2):
        ker = np.ones((3,3), np.uint8)
        g = cv2.morphologyEx(self.grid, cv2.MORPH_CLOSE, ker, iterations=1)
        g = np.flipud(g)  # Y arriba
        return cv2.resize(g, (self.W*scale, self.H*scale), interpolation=cv2.INTER_NEAREST)

def run_live(input_src="0", cell=0.15, grid_m=30.0, kf_stride=3,
             max_feats=1500, ratio=0.75, ransac_th=2.0, reproj_th=4.0,
             show=True, save_every=30):
    os.makedirs("img-code", exist_ok=True)

    cap = open_source(str(input_src))
    ok, f0 = cap.read(); 
    if not ok: raise RuntimeError("No pude leer primer frame.")
    gray_prev = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    K = guess_K(f0.shape)
    orb = cv2.ORB_create(max_feats)

    # Pose world (Rcw, tcw)
    Rcw_prev, tcw_prev = np.eye(3), np.zeros((3,1))
    # KF de arranque
    kf_gray, Rcw_kf, tcw_kf = gray_prev.copy(), Rcw_prev.copy(), tcw_prev.copy()

    og = OccGrid(cell=cell, grid_m=grid_m)
    frame_idx, saved = 0, 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_idx += 1

        matches, kp1, kp2 = orb_knn(gray_prev, gray, orb, ratio)
        if len(matches) >= 8:
            pts1, pts2 = pts_from_matches(kp1, kp2, matches)
            E, maskE = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=ransac_th)
            if E is not None and maskE is not None and int(maskE.sum()) >= 8:
                _, R, t, maskP = cv2.recoverPose(E, pts1, pts2, K)
                # acumular pose
                Rcw_cur, tcw_cur = compose(R, t, Rcw_prev, tcw_prev)

                # cada kf_stride triangulamos KF vs actual y actualizamos grid
                if frame_idx % kf_stride == 0:
                    m2, kp_kf, kp_c = orb_knn(kf_gray, gray, orb, ratio)
                    if len(m2) >= 20:
                        uv_kf, uv_c = pts_from_matches(kp_kf, kp_c, m2)
                        X = triangulate_two(K, Rcw_kf, tcw_kf, Rcw_cur, tcw_cur, uv_kf, uv_c)
                        err1, Xc1 = reproj_err(K, Rcw_kf,  tcw_kf,  X, uv_kf)
                        err2, Xc2 = reproj_err(K, Rcw_cur, tcw_cur, X, uv_c)
                        valid = (Xc1[:,2] > 0) & (Xc2[:,2] > 0) & (err1 < reproj_th) & (err2 < reproj_th)
                        P = X[valid]
                        # usamos planta XZ (escala relativa)
                        XY = P[:, [0, 2]]
                        og.update(XY)

                        # recenter grid alrededor de la cámara (en XZ)
                        cam_xyz = (-Rcw_cur.T @ tcw_cur).ravel()
                        og.recenter_if_needed(np.array([cam_xyz[0], cam_xyz[2]]))

                        # actualizar KF al actual
                        kf_gray, Rcw_kf, tcw_kf = gray.copy(), Rcw_cur.copy(), tcw_cur.copy()

                Rcw_prev, tcw_prev = Rcw_cur, tcw_cur

        gray_prev = gray

        if show:
            vis = og.vis(scale=3)
            cv2.imshow("Grid (ESC para salir)", vis)
            # opcional, ver matches rápidos
            # if len(matches) > 0:
            #     showm = cv2.drawMatches(cv2.cvtColor(gray_prev,cv2.COLOR_GRAY2BGR), kp1,
            #                             cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR), kp2,
            #                             matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            #     cv2.imshow("Matches", showm)

        if save_every > 0 and frame_idx % save_every == 0:
            out = os.path.join("img-code", f"grid_live_{saved:05d}.png")
            cv2.imwrite(out, og.vis(scale=2)); saved += 1

        if show and (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release(); cv2.destroyAllWindows()
    final_png = os.path.join("img-code", "grid_live_final.png")
    cv2.imwrite(final_png, og.vis(scale=2))
    print(f"✅ Guardado snapshot final: {final_png}")

def parse_args():
    ap = argparse.ArgumentParser("Live VO Mapper")
    ap.add_argument("--input", default=None, help="ruta de video; si no, usa --camera")
    ap.add_argument("--camera", default="0", help="índice de cámara (0,1,...)")
    ap.add_argument("--cell", type=float, default=0.15)
    ap.add_argument("--grid_m", type=float, default=30.0)
    ap.add_argument("--kf_stride", type=int, default=3)
    ap.add_argument("--max_feats", type=int, default=1500)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--ransac_th", type=float, default=2.0)
    ap.add_argument("--reproj_th", type=float, default=4.0)
    ap.add_argument("--no_show", action="store_true")
    ap.add_argument("--save_every", type=int, default=30, help="guardar snapshot cada N frames (0=off)")
    return ap.parse_args()

if __name__ == "__main__":
    a = parse_args()
    src = a.input if a.input else a.camera
    run_live(src, a.cell, a.grid_m, a.kf_stride, a.max_feats, a.ratio, a.ransac_th, a.reproj_th,
             show=not a.no_show, save_every=a.save_every)
