# main.py
import argparse
from vo import run_vo
from plot_traj import plot_traj_cli
from vo_triangulate import triangulate_cli

def main():
    ap = argparse.ArgumentParser("Runner VO-BRIGITTE")
    ap.add_argument("cmd", choices=["run"], help="run")
    ap.add_argument("--input", required=True)             # mp4/MOV o índice de cámara (0/1)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--no-triang", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    # 1) VO -> trajectory.npy
    traj_path = "trajectory.npy"
    print("▶ VO…")
    run_vo(args.input, out_path=traj_path, max_frames=args.max_frames, display=False)

    # 2) Plot
    img = None
    if not args.no_plot:
        print("▶ Plot…")
        img = plot_traj_cli(traj_path, "img-code/traj.png")

    # 3) Triangulación (sparse)
    ply = None
    if not args.no_triang:
        print("▶ Triangulación…")
        ply = triangulate_cli(args.input, "points.ply", max_frames=args.max_frames)

    print("✅ DONE")
    print({"trajectory": traj_path, "plot": img, "points": ply})

if __name__ == "__main__":
    main()
