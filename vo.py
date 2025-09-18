# vo.py
import argparse, cv2, numpy as np, os, sys

def open_source(src):
    """Permite índice de cámara ('0','1') o ruta a archivo."""
    if isinstance(src, str) and src.isdigit():
        cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)  # DSHOW mejora en Windows
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente: {src}")
    return cap

def estimate_intrinsics(frame_shape):
    """Intrínsecos aproximados (sirven para demo/pipe)."""
    h, w = frame_shape[:2]
    fx = fy = 0.9 * w   # foco aprox (ajustable)
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    return K

def run_vo(input_src, out_path="trajectory.npy", max_frames=None, display=True):
    cap = open_source(input_src)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("No se pudo leer el primer frame")
    K = estimate_intrinsics(frame.shape)

    orb = cv2.ORB_create(2000)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

    # Pose acumulada (R,t). Iniciamos en el origen
    R_cum = np.eye(3)
    t_cum = np.zeros((3, 1))

    traj = [t_cum.ravel().copy()]  # lista de posiciones (N,3)

    frame_count = 0
    while True:
        if max_frames is not None and frame_count >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)

        if prev_des is not None and des is not None and len(prev_des) > 0 and len(des) > 0:
            matches = bf.match(prev_des, des)
            if len(matches) >= 8:
                matches = sorted(matches, key=lambda m: m.distance)[:200]

                pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp[m.trainIdx].pt   for m in matches])

                # Matriz esencial + recuperación de pose
                E, mask = cv2.findEssentialMat(pts2, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is not None:
                    _, R, t, mask_pose = cv2.recoverPose(E, pts2, pts1, K)

                    # Escala arbitraria (1.0). Para demo/pipeline.
                    scale = 1.0
                    t_step = (R_cum @ t) * scale
                    t_cum = t_cum + t_step
                    R_cum = R @ R_cum

                    traj.append(t_cum.ravel().copy())

                    if display:
                        draw = cv2.drawMatches(
                            cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR), prev_kp,
                            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), kp,
                            matches[:50], None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                        )
                        cv2.imshow("Matches (ESC para salir)", draw)
                        if cv2.waitKey(1) & 0xFF == 27:
                            break

        prev_gray, prev_kp, prev_des = gray, kp, des
        frame_count += 1

    cap.release()
    if display:
        cv2.destroyAllWindows()

    traj = np.vstack(traj) if len(traj) else np.zeros((0,3), dtype=np.float32)
    np.save(out_path, traj.astype(np.float32))
    print(f"✅ Guardado {out_path} con {len(traj)} poses")

def parse_args():
    ap = argparse.ArgumentParser(description="Visual Odometry simple (demo)")
    ap.add_argument("--input", required=False, default="test1.mp4",
                    help="Ruta a video (.mp4/.MOV) o índice de cámara (0,1)")
    ap.add_argument("--out", required=False, default="trajectory.npy",
                    help="Archivo de salida .npy")
    ap.add_argument("--max_frames", type=int, default=None,
                    help="Límite de frames para test")
    ap.add_argument("--no-display", action="store_true",
                    help="No mostrar ventana de matches")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_vo(
        input_src=str(args.input),
        out_path=args.out,
        max_frames=args.max_frames,
        display=not args.no_display
    )
