# VO-Básico 

Un intento de aprender slam. La idea es estimar el movimiente de una cámara (odometría visual) y dibujar esa trayectoria. Luego se guarda la "información" (matrices) del recorrido y se realiza un grafico.

> Nota: Al ser **monocular** (única camara y por ende único input), la **escala es relativa** (no tenés unidades reales sin una referencia externa).

## Estructura

VO-basic/
├─ vo.py # demo de matches ORB (visualización)
├─ vo_pose.py # odometría: R,t + trayectoria 2D + guardado .npy
├─ plot_traj.py # script para graficar la trayectoria guardada
├─ ort.MOV # video de prueba
├─ test1.mp4  # video de prueba, este es con el que mejor  anda
├─ requirements.txt
└─ .gitignore

## Requisitos

- Python 3.9+ (probado en macOS)
- Paquetes: ver `requirements.txt` (OpenCV, NumPy, Matplotlib, etc.)

## Quickstart

```bash

python3 -m venv .venv
source .venv/bin/activate


pip install -r requirements.txt

# 3) Correr con video
python3 vo_pose.py --video ort.MOV --show-matches

#    o con webcam (ej. cámara 0)
python3 vo_pose.py --camera 0 --show-matches

# 4) en esta etapa se va a crear un archivo .npy y se puede graficar la trayectoria 
python3 plot_traj.py
```

![img 1](readme-images/1.png)
![img 2](readme-images/2.png)