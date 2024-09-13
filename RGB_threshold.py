import cv2
import numpy as np

def nothing(x):
    pass

# 创建一个窗口
cv2.namedWindow('Live')

# 创建RGB阈值滑动条
cv2.createTrackbar('RMin', 'Live', 0, 255, nothing)
cv2.createTrackbar('RMax', 'Live', 0, 255, nothing)
cv2.createTrackbar('GMin', 'Live', 0, 255, nothing)
cv2.createTrackbar('GMax', 'Live', 0, 255, nothing)
cv2.createTrackbar('BMin', 'Live', 0, 255, nothing)
cv2.createTrackbar('BMax', 'Live', 0, 255, nothing)

# 初始化摄像头
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

while cap.isOpened():
    retval, frame = cap.read()
    
    if not retval:
        break

    # 获取当前RGB阈值
    rMin = cv2.getTrackbarPos('RMin', 'Live')
    rMax = cv2.getTrackbarPos('RMax', 'Live')
    gMin = cv2.getTrackbarPos('GMin', 'Live')
    gMax = cv2.getTrackbarPos('GMax', 'Live')
    bMin = cv2.getTrackbarPos('BMin', 'Live')
    bMax = cv2.getTrackbarPos('BMax', 'Live')

    # 设置RGB范围
    lower_red = np.array([bMin, gMin, rMin])
    upper_red = np.array([bMax, gMax, rMax])
    
    # 创建掩膜
    mask = cv2.inRange(frame, lower_red, upper_red)
    
    # 将掩膜应用于原始图像
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # 二值化
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 显示图像
    cv2.imshow('Live', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Binary', binary)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
