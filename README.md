# VO-Básico 

Un intento de aprender SLAM monocular.
La idea es:

Estimar el movimiento de una cámara (odometría visual) y dibujar su trayectoria.

Guardar la información (matrices y puntos 3D) del recorrido.

Triangular una nube de puntos 3D.

Generar un mapa 2D tipo “aspiradora” (Occupancy Grid) desde los puntos.

> Nota: Al ser **monocular** (única camara y por ende único input), la **escala es relativa** (no tenés unidades reales sin una referencia externa).

## Estructura

vo-brigitte/
├─ vo.py                # Demo de matches ORB (visualización rápida)
├─ vo_pose.py           # Odometría: calcula R,t + trayectoria 2D + guarda trajectory.npy
├─ plot_traj.py         # Script para graficar la trayectoria guardada
├─ vo_triangulate.py    # Triangula nube de puntos 3D desde la trayectoria → points.npy (+ colors.npy)
├─ viz_points.py        # Visualiza la nube 3D con Open3D
├─ gridmap_from_points.py # Genera un mapa 2D (Occupancy Grid) desde points.npy
├─ test1.mp4            # Video de prueba principal
├─ ort.MOV              # Otro video de prueba
├─ requirements.txt    
├─ README.md
└─ .gitignore


## Requisitos

- Python 3.9+ (probado en macOS)
- Paquetes: ver `requirements.txt` (OpenCV, NumPy, Matplotlib, etc.)

## Quickstart

```bash
 # en mac
python3 -m venv .venv
source .venv/bin/activate


pip install -r requirements.txt

# 3) Correr con video
python3 vo_pose.py --video ort.MOV --show-matches

#    o con webcam (ej. cámara 0)
python3 vo_pose.py --camera 0 --show-matches

# 4) en esta etapa se va a crear un archivo .npy y se puede graficar la trayectoria 
python3 plot_traj.py

#5 genera points.npy 
python vo_triangulate.py
#6 ver la nube 3d 
python viz_points.py
#generar mapa 
python gridmap_from_points.py --use_open3d --cell 0.05 --height_thresh 0.15 --downsample 2

```

![img 1](readme-images/1.png)
![img 2](readme-images/2.png)