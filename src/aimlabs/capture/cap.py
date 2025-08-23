import dxcam
import cv2

camera = dxcam.create()  # 默认主显示器
camera.start(target_fps=60)  # 启动捕获

cv2.namedWindow("Screen", cv2.WINDOW_NORMAL) 

while True:
    frame = camera.get_latest_frame()  # 获取最新帧（非阻塞）
    if frame is not None:
        # OpenCV 处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Screen", gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.stop()
