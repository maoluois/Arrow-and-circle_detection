import cv2
import numpy as np

def nothing(x):
    pass

# 创建一个窗口
cv2.namedWindow('Live')

# 创建LAB阈值滑动条
cv2.createTrackbar('LMin', 'Live', 0, 100, nothing)
cv2.createTrackbar('LMax', 'Live', 100, 100, nothing)
cv2.createTrackbar('AMin', 'Live', -128, 128, nothing)
cv2.createTrackbar('AMax', 'Live', 128, 128, nothing)
cv2.createTrackbar('BMin', 'Live', -128, 128, nothing)
cv2.createTrackbar('BMax', 'Live', 128, 128, nothing)

# 初始化摄像头
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

while cap.isOpened():
    retval, frame = cap.read()
    
    if not retval:
        break

    # 转换到LAB色彩空间
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

    # 获取当前LAB阈值
    lMin = cv2.getTrackbarPos('LMin', 'Live')
    lMax = cv2.getTrackbarPos('LMax', 'Live')
    aMin = cv2.getTrackbarPos('AMin', 'Live') - 128
    aMax = cv2.getTrackbarPos('AMax', 'Live') - 128
    bMin = cv2.getTrackbarPos('BMin', 'Live') - 128
    bMax = cv2.getTrackbarPos('BMax', 'Live') - 128

    # 设置LAB范围
    lower_lab = np.array([lMin, aMin, bMin], dtype="uint8")
    upper_lab = np.array([lMax, aMax, bMax], dtype="uint8")
    
    # 创建掩膜
    mask = cv2.inRange(lab_frame, lower_lab, upper_lab)
    
    # 将掩膜应用于原始图像
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # 二值化
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) 
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 显示图像
    cv2.imshow('Live', frame)
    cv2.imshow("Gray", gray)
    # cv2.imshow('Mask', mask)
    # cv2.imshow('Binary', binary)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()