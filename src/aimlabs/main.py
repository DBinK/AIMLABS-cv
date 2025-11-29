import dxcam
import cv2

from looptick import LoopTick


loop = LoopTick()
camera = dxcam.create(0, region=(0,0,1280,720))  # 默认主显示器
camera.start(target_fps=60, video_mode=True)  # 启动捕获

# cv2.namedWindow("Screen", cv2.WINDOW_NORMAL) 

while True:
    frame = camera.get_latest_frame()  # 获取最新帧（非阻塞）
    if frame is not None:
        loop.tick()
        hz = loop.get_avg_hz()
        print(hz)
        # pose_kpts = mp.detect_pose(frame)
        
        cv2.imshow("Screen", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
