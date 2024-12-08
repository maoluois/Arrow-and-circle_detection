import cv2
import numpy as np
import serial

# 初始化前一帧的坐标S
prev_x, prev_y = None, None

# 设置低通滤波器的alpha参数，介于0和1之间
alpha = 0.8

def find_cir(gray, color):
    global alpha, prev_x, prev_y

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    # 添加圆形检测
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=100, param2=43, minRadius=90, maxRadius=120)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if prev_x is None or prev_y is None:
                prev_x, prev_y = x, y
            else:
                # 使用低通滤波器平滑坐标
                x = int(alpha * x + (1 - alpha) * prev_x)
                y = int(alpha * y + (1 - alpha) * prev_y)
                prev_x, prev_y = x, y
            # 绘制圆轮廓
            cv2.circle(frame, (prev_x, prev_y), r, (0, 255, 0), 2)
            # 绘制圆心
            cv2.circle(frame, (prev_x, prev_y), 2, (0, 0, 255), -1)
        
        print(f"{color}x:{prev_x},y:{prev_y}")
    
    cv2.imshow("Detected Circles", frame)

def find_cir0(gray, color):
    global alpha, prev_x, prev_y

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    # 添加圆形检测
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=100, param2=43, minRadius=90, maxRadius=120)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if prev_x is None or prev_y is None:
                prev_x, prev_y = x, y
            else:
                # 使用低通滤波器平滑坐标
                x = int(alpha * x + (1 - alpha) * prev_x)
                y = int(alpha * y + (1 - alpha) * prev_y)
                prev_x, prev_y = x, y
            # 绘制圆轮廓
            cv2.circle(frame, (prev_x, prev_y), r, (0, 255, 0), 2)
            # 绘制圆心
            cv2.circle(frame, (prev_x, prev_y), 2, (0, 0, 255), -1)
        
        print(f"{color}x:{prev_x},y:{prev_y}")
    
    cv2.imshow("Detected Circles", frame)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

while cap.isOpened():
    retval, frame = cap.read()
    if not retval:
        print("无法读取帧")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green =  np.array([48, 67, 131])
    upper_green = np.array([79, 197, 253])

    lower_blue = np.array([21, 79, 56])
    upper_blue = np.array([135, 157, 137])
   
    lower_red1 = np.array([0, 156, 67])
    upper_red1 = np.array([179, 255, 255])
    lower_red2 = np.array([164, 128, 64])
    upper_red2 = np.array([179, 255, 255])

    lower_green0 =  np.array([48, 67, 131])
    upper_green0 = np.array([79, 197, 253])

    lower_blue0 = np.array([21, 79, 56])
    upper_blue0 = np.array([135, 157, 137])
   
    lower_red10 = np.array([0, 120, 70])
    upper_red10 = np.array([10, 255, 255])
    lower_red20 = np.array([170, 120, 70])
    upper_red20 = np.array([180, 255, 255])

    # 创建红色掩膜
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    r_mask = mask1 + mask2
    r_regions = cv2.bitwise_and(frame, frame, mask=r_mask)

    mask3 = cv2.inRange(hsv, lower_red10, upper_red10)
    mask4 = cv2.inRange(hsv, lower_red20, upper_red20)
    r0_mask = mask3 + mask4
    r0_regions = cv2.bitwise_and(frame, frame, mask=r0_mask)

    g_mask = cv2.inRange(hsv, lower_green, upper_green)
    g_regions = cv2.bitwise_and(frame, frame, mask=g_mask)

    g0_mask = cv2.inRange(hsv, lower_green0, upper_green0)
    g0_regions = cv2.bitwise_and(frame, frame, mask=g0_mask)

    b_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    b_regions = cv2.bitwise_and(frame, frame, mask=b_mask)

    b0_mask = cv2.inRange(hsv, lower_blue0, upper_blue0)
    b0_regions = cv2.bitwise_and(frame, frame, mask=b0_mask)

    r_gray = cv2.cvtColor(r_regions, cv2.COLOR_BGR2GRAY)
    r0_gray = cv2.cvtColor(r0_regions, cv2.COLOR_BGR2GRAY)
    g_gray = cv2.cvtColor(g_regions, cv2.COLOR_BGR2GRAY)
    g0_gray = cv2.cvtColor(g0_regions, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b_regions, cv2.COLOR_BGR2GRAY)
    b0_gray = cv2.cvtColor(b0_regions, cv2.COLOR_BGR2GRAY)

    cv2.imshow("r_gray", r_gray)

    cv2.imshow("g_gray", g_gray)
    
    cv2.imshow("b_gray", b_gray)

    cv2.imshow("r0_gray", r0_gray)

    cv2.imshow("g0_gray", g0_gray)
    
    cv2.imshow("b0_gray", b0_gray)

    find_cir(r_gray, "红")
    
    find_cir(g_gray, "绿")
    
    find_cir(b_gray, "蓝")

    find_cir0(r0_gray, "红环")
    
    find_cir0(g0_gray, "绿环")
    
    find_cir0(b0_gray, "蓝环")


   
   
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q退出
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()