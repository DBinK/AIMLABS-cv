# import dxcam
from pydirectinput import click
from numpy import array
import dxcam_cpp as dxcam
from looptick import LoopTick

i = 1

WIDTH = 3840
HEIGHT = 2160

loop = LoopTick()
camera = dxcam.create(
    0, region=(int(WIDTH/2), int(HEIGHT/2), int(WIDTH/2 + i), int(HEIGHT/2 + i),)
)
camera.start(target_fps=120)  # 启动捕获

red = array([[[230, 55, 55]]])

last_frame = array([[[0, 0, 0]]])

mx = int(WIDTH/2)
my = int(HEIGHT/2)

while True:
    frame = camera.grab()
    if frame is not None:
        loop.tick()
        hz = loop.get_avg_hz()
        print(hz)

        if (last_frame == red).any() and (frame != red).any():
            click(mx, my)
            
        last_frame = frame
        