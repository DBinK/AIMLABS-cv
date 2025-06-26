import cv2 
import numpy as np

CYAN_LOWER = [78, 43, 46]
CYAN_UPPER = [99, 255, 255]
# CYAN_LOWER = [100, 43, 46]
# CYAN_UPPER = [124, 255, 255]

BLUE_LOWER = [100, 43, 46]   
BLUE_UPPER = [124, 255, 255]

def pre_process(img):
    """ 预处理图像 """
    img_blur = cv2.GaussianBlur(img, (25, 25), 0)
    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)  # 转换为HSV格式
    return img_blur, img_hsv

def remove_background(img_hsv, lower, upper):
    """ 创建颜色掩膜 """ 
    lower_color = np.array(lower)  # 颜色范围下限
    upper_color = np.array(upper)  # 颜色颜色上限

    mask = cv2.inRange(img_hsv, lower_color, upper_color)  # 创建掩膜
    cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
    cv2.imshow('Mask', mask)

    # kernel = np.ones((5, 5), np.uint8)             # 定义结构元素  
    # mask = cv2.erode(mask, kernel, iterations=5)   # 收缩操作
    # mask = cv2.dilate(mask, kernel, iterations=5)  # 膨胀操作
 
    # cv2.bitwise_not(mask, mask)  # 翻转掩膜 

    return mask

def detect_circle_contours(img):
    """ 从背景识别圆形目标轮廓 """
    mask_cyan = remove_background(img, CYAN_LOWER, CYAN_UPPER)  # 提取圆形目标掩膜
    contours, _ = cv2.findContours(mask_cyan, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
    
    filtered_contours = []

    print(f"轮廓数量: {len(contours)}")

    for contour in contours:  # 提取位置和颜色

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter < (2 * 4) or area < 5:  # 过滤小的轮廓 
            continue
        
        # 计算长宽比
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        # print(f"长宽比: {aspect_ratio}")
        
        # 计算 最小外接圆 与 轮廓 的面积之比
        (x, y), radius = cv2.minEnclosingCircle(contour)
        min_circle_area = np.pi * radius ** 2
        area_ratio = min_circle_area / area
        # cv2.circle(img, (int(x), int(y)), int(radius), (0, 0, 255), 2) # 绘制最小外接圆
        # print(f"最小外接圆面积与轮廓面积之比: {area_ratio}")
        
        # 过滤条件
        if (area_ratio < 1.3)  and (0.90 < aspect_ratio < 1.10):
            filtered_contours.append(contour)

        # filtered_contours.append(contour)

    return filtered_contours     

def contours_to_positom(contours):
    """ 将轮廓转换为圆形目标位置 """

    contours_positions = []

    for contour in contours:
        M = cv2.moments(contour)  # 计算轮廓的中心点
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            contours_positions.append((cX, cY))

    return contours_positions

def detect_circle(img):
    """ 识别圆形目标总函数 """
    contours = detect_circle_contours(img)
    pos = contours_to_positom(contours)
    return pos

def draw_circle(img, contours, color):
    """ 在图像上绘制圆形目标轮廓 """

    img_circle = img.copy()

    # for contour in contours:
    #     cv2.drawContours(img_circle, [contour], -1, color, 8)

    circle_positions = contours_to_positom(contours)
    for circle_position in circle_positions:
        cv2.circle(img_circle, circle_position, 4, color, -1)
        cv2.putText(img_circle, f" {circle_position[0], circle_position[1]}", circle_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return img_circle

#################################################################
def get_point_color(img, center_point, radius):
    """ 获取指定点周围区域的平均颜色 """

    x, y = center_point     # 定义周围区域的坐标
    x1 = max(0, x - radius)
    x2 = min(img.shape[1], x + radius)
    y1 = max(0, y - radius)
    y2 = min(img.shape[0], y + radius)
    
    region = img[y1:y2, x1:x2]            # 获取周围区域并计算平均颜色
    average_color = cv2.mean(region)[:3]  # 获取 BGR 平均值并忽略 alpha 通道
 
    average_color = tuple(int(c) for c in average_color)  # 对平均颜色进行取整

    # cv2.circle(img, center_point, radius, (255, 0, 0), 2)  # 检查指定点采样半径用

    return average_color
def get_crosshair(img: np.ndarray) -> tuple[int, int]:
    """返回固定位置的准星坐标（适用于准星位置不变的游戏）"""
    height, width = img.shape[:2]
    return (width // 2, height // 2)  # 假设准星在屏幕中央

def is_point_in_range(img_hsv: np.ndarray, x: int, y: int) -> bool:
    """检查图像中 (x, y) 处的像素是否在 CYAN 范围内"""
    pixel_hsv = img_hsv[y, x]  # 注意 OpenCV 是 (y, x) 顺序

    lower = np.array(CYAN_LOWER)  # 转换为 numpy 数组
    upper = np.array(CYAN_UPPER)

    return bool(cv2.inRange(np.array([pixel_hsv]), lower, upper).any())

def is_shoot(img_hsv: np.ndarray) -> bool: 
    crosshair = get_crosshair(img)
    if is_point_in_range(img_hsv, crosshair[0], crosshair[1]):
        return  True
    return False


if __name__ == '__main__':
    img = cv2.imread('test.png')
    img_blur, img_hsv = pre_process(img)
    contours = detect_circle_contours(img_blur)
    pos = contours_to_positom(contours)
    print(pos)

    img_circle = draw_circle(img, contours, (0, 10, 200))

    cv2.namedWindow("img_circle", cv2.WINDOW_NORMAL)
    cv2.imshow("img_circle", img_circle)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    
    cv2.namedWindow('img_blur', cv2.WINDOW_NORMAL)
    cv2.imshow('img_blur', img_blur)

    # print(f"是否开火: {is_shoot(img_hsv)}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

