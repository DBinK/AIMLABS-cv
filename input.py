import pyautogui
import pydirectinput

pydirectinput.moveTo(100, 150) # 将鼠标移动到坐标(100, 150)的位置

pydirectinput.click() # 在鼠标当前位置单击
pydirectinput.click(200, 220) # 在坐标(200, 220)的位置单击鼠标
pydirectinput.doubleClick() # 在当前位置双击鼠标

pydirectinput.move(None, 10)  # 将鼠标向下移动10像素（相对于当前位置移动）

pydirectinput.press('esc') # 模拟按下ESC键
pydirectinput.keyDown('shift') # 按下shift键 
pydirectinput.keyUp('shift') # 松开shift键
