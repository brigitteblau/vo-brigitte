import cv2
import numpy as np
import argparse
import os

def parse_args():
    p = argparse.ArgumentParser(description="VO monocular: ORB + Essential + recoverPose + trayectoria")
    p.add_argument("--video", type=str, default=None, help="Ruta al video (ej: ./ort.MOV). Si no se pasa, usa webcam.")
    p.add_argument("--camera", type=int, default=0, help="Índice de cámara (0 por defecto).")
    p.add_argument("--max-feats", type=int, default=2000, help="Cantidad de features ORB")
    p.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio test")
    p.add_argument("--ransac-th", type=float, default=1.0, help="Umbral RANSAC p/ Essential (px)")
    p.add_argument("--show-matches", action="store_true", help="Mostrar ventana con matches")
    return p.parse_args()

def open_capture(args):
    if args.video:
        if not os.path.exists(args.video):
            raise FileNotFoundError(f"No se encontró el video: {args.video}")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened(): raise RuntimeError(f"No se pudo abrir el video: {args.video}")
        print(f"[OK] Video: {args.video}")
        return cap
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara. Revisá permisos de Cámara en macOS.")
    print(f"[OK] Cámara índice {args.camera}")
    return cap

def guess_K(frame_shape):
    h, w = frame_shape[:2]
    focal = 0.9 * max(w, h)             # estimación grosera
    K = np.array([[focal,   0.0, w/2.0],
                  [  0.0, focal, h/2.0],
                  [  0.0,   0.0,  1.0]], dtype=np.float64)
    return K

def orb_matches(img1, img2, orb, ratio=0.75):
    # KNN + Lowe ratio test
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None or len(des1)==0 or len(des2)==0:
        return [], kp1, kp2
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des1, des2, k=2)
    good = []
    for m,n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good, kp1, kp2

def points_from_matches(kp1, kp2, matches):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2

def draw_trajectory(canvas, traj_pts):
    canvas[:] = 255
    if len(traj_pts) < 2: 
        return canvas
    # normalizar a canvas
    pts = np.array(traj_pts, dtype=np.float32)
    # centrar y escalar automáticamente
    minxy = pts.min(axis=0)
    maxxy = pts.max(axis=0)
    span = np.maximum(maxxy - minxy, 1e-3)
    margin = 40
    w, h = canvas.shape[1], canvas.shape[0]
    scale = 0.9 * min((w-2*margin)/span[0], (h-2*margin)/span[1])
    pts_n = (pts - minxy) * scale + np.array([margin, margin])
    for i in range(1, len(pts_n)):
        p1 = tuple(np.int32(pts_n[i-1]))
        p2 = tuple(np.int32(pts_n[i]))
        cv2.line(canvas, p1, p2, (0,0,0), 2, cv2.LINE_AA)
    cv2.circle(canvas, tuple(np.int32(pts_n[-1])), 4, (0,0,255), -1)
    cv2.putText(canvas, f"Frames: {len(traj_pts)}", (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return canvas

def main():
    args = parse_args()
    cap = open_capture(args)

    # Primer frame para definir K
    ok, frame_prev = cap.read()
    if not ok:
        print("No pude leer el primer frame.")
        return
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    K = guess_K(frame_prev.shape)
    Kinv = np.linalg.inv(K)

    orb = cv2.ORB_create(args.max_feats)

    # Pose acumulada (world = primer frame)
    R_w = np.eye(3, dtype=np.float64)
    t_w = np.zeros((3,1), dtype=np.float64)
    traj_2d = []   # guardamos (x,z) (aprox.) o (x,y). Usaremos (x,z) convencion “hacia adelante = z”.

    # canvas para trayectoria
    traj_canvas = np.ones((600, 600, 3), dtype=np.uint8) * 255

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[INFO] Fin de video.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # matches
        matches, kp1, kp2 = orb_matches(gray_prev, gray, orb, ratio=args.ratio)

        if len(matches) >= 8:
            pts1, pts2 = points_from_matches(kp1, kp2, matches)

            # Estimar Essential con RANSAC
            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=args.ransac_th)
            if E is not None:
                inliers = mask.ravel().sum() if mask is not None else 0

                if inliers >= 8:
                    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

                    # Actualizar pose mundial:
                    # R_w_new = R * R_w
                    # t_w_new = t_w + R_w * t   (t está en coords de cámara previa)
                    R_w = R @ R_w
                    t_w = t_w + (R_w @ t)

                    # Guardar punto 2D (x,z) para dibujar (monocular = escala relativa)
                    x, z = float(t_w[0]), float(t_w[2])
                    traj_2d.append((x, z))

                    # Dibujar trayectoria
                    traj_canvas = draw_trajectory(traj_canvas, traj_2d)
                    cv2.imshow("Trayectoria 2D (escala relativa)", traj_canvas)

                    # Opcional: mostrar matches inliers
                    if args.show_matches:
                        inlier_matches = [m for i,m in enumerate(matches) if mask[i]]
                        vis = cv2.drawMatches(frame_prev, kp1, frame, kp2, inlier_matches, None,
                                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                        cv2.putText(vis, f"Inliers: {inliers}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        cv2.imshow("Matches (inliers)", vis)
                else:
                    # Muy pocos inliers, solo mostrar frame
                    if args.show_matches:
                        cv2.imshow("Matches (inliers)", frame)
            else:
                if args.show_matches:
                    cv2.imshow("Matches (inliers)", frame)
        else:
            if args.show_matches:
                cv2.imshow("Matches (inliers)", frame)

        gray_prev = gray
        frame_prev = frame

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    np.save("trajectory.npy", np.array(traj_2d))
    print("[OK] Trayectoria guardada en trajectory.npy")

if __name__ == "__main__":
    main()
