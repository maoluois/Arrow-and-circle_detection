import cv2
import numpy as np
template_image = cv2.imread('/home/lmz/opencv/WB.jpg')  # 

# 将图像从BGR转换到HSV色彩空间
template_image_hsv = cv2.cvtColor(template_image, cv2.COLOR_BGR2HSV)    
lower_red1 = np.array([0, 156, 67])
upper_red1 = np.array([179, 255, 255])  # 截图尝试
lower_red2 = np.array([164, 128, 64])
upper_red2 = np.array([179, 255, 255])
# 创建红色掩膜
mask1 = cv2.inRange(template_image_hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(template_image_hsv, lower_red2, upper_red2)
mask = mask1 + mask2

# 对原图像和掩膜进行位运算
red_regions = cv2.bitwise_and(template_image_hsv, template_image_hsv, mask=mask)

# 转换为灰度图像
gray = cv2.cvtColor(red_regions, cv2.COLOR_BGR2GRAY)

# 使用高斯模糊平滑图像
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 形态学闭操作
new_image = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

cv2.imshow('new_image', new_image)

# 应用边缘检测
template_edges = cv2.Canny(new_image, 50, 150)

# 寻找轮廓
template_contours, _ = cv2.findContours(template_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sorted_template_contours = sorted(template_contours, key=cv2.contourArea, reverse=True)
for contour in sorted_template_contours:

    # 计算轮廓的周长
    perimeter = cv2.arcLength(contour, True)
    
    # 多边形拟合
    approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
    # cv2.drawContours(template_image, contour, -1, (255, 0, 0), 2)
    print(cv2.contourArea(approx))
    # if cv2.contourArea(approx) > 2555.0:
    #     cv2.drawContours(template_image, [approx], -1, (0, 255, 0), 2)
    # print(approx)
    # 如果多边形有四个顶点，则认为它是一个四边形
    
    if len(approx) == 4:
        # print(cv2.contourArea(approx))
        cv2.drawContours(template_image, [approx], -1, (0, 255, 0), 2)
        # print(points)
    


cv2.imshow('template_image', template_image)
cv2.waitKey(0)
cv2.destroyAllWindows()