import cv2
import numpy as np

# 全局变量，用于存储前几帧的质心位置
centroids = []
alpha = 0.5  # 滤波因子，控制新值和历史值的权重
last_cx , last_cy = 0, 0
last_vertex = (0, 0)
test = []

def Lowpass(alpha, now, last):
    return alpha * now + (1 - alpha) * last

def find_centroid(contour, f, max=10):

    global centroids, alpha, last_cx, last_cy
    
    M = cv2.moments(contour)
    # print(f"矩: {M}")

    if M["m00"] != 0:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
    else:
        cX, cY = 0, 0
        print("无法计算质心")
    
    # 均值和低通滤波不好用
    # centroids.append((cX, cY))
    # if len(centroids) > max:
    #     centroids.pop(0)
    # else:
    #     return
    # mean_cX = int(sum([cX for cX, cY in centroids]) / len(centroids))
    # mean_cY = int(sum([cY for cX, cY in centroids]) / len(centroids))

    last_cx, last_cy = cX, cY
    
    # if abs(cX - last_cx) > 50 or abs(cY - last_cy) > 50:
    #     cX, cY = last_cx, last_cy
    # print("质心偏差过大，使用上一帧的质心")

    
    # last_cx, last_cy = mean_cX, mean_cY
    
    # if abs(cX - last_cx) > 20 or abs(cY - last_cy) > 20:
    #     mean_cX, mean_cY = last_cx, last_cy
    # print("质心偏差过大，使用上一帧的质心")

    cv2.circle(f, (cX, cY), 5, (0, 0, 255), -1)

    # 在帧上显示质心坐标
    cv2.putText(f, f"Centroid: ({cX}, {cY})", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    print(f"质心坐标为：({cX}, {cY})")
    return (cX, cY)
def calculate_angle_point(point1, point2, vertex):
    # 计算角度
    line1 = np.array(point1) - np.array(vertex)
    line2 = np.array(point2) - np.array(vertex)
    
    dot_product = np.dot(line1, line2)
    
    norm_line1 = np.linalg.norm(line1)
    norm_line2 = np.linalg.norm(line2)
    
    angle_rad = np.arccos(dot_product / (norm_line1 * norm_line2))
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def calculate_angle_line(line1, line2):
    # 计算角度
    dot_product = np.dot(line1, line2)
    
    norm_line1 = np.linalg.norm(line1)
    norm_line2 = np.linalg.norm(line2)
    
    angle_rad = np.arccos(dot_product / (norm_line1 * norm_line2))
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def find_vertex(points):
    global last_vertex, test
    vertex = (0, 0)
    n = len(points) 
    points = np.array(points)
    for k in range(n):
        vertex = points[k]
        
        # vectors = points - vertex # 相当于每个数组里每个元素减vertex

        for i in range(n):
            if i == k:
                continue

            for j in range(i + 1, n):
                if j == k:
                    continue

                angle = calculate_angle_point(points[i], points[j], vertex)
                test.append(angle)
                
                if 88 <= angle <= 92:
                    # 低通滤波？
                    print(f"角度判据为{angle}")
                    last_vertex = vertex
                    return vertex 
                else:
                    continue
                    
    return last_vertex  

def get_angle(centre_line, arrow_centre_line):
    angle_360 = 0
    angle_180 = calculate_angle_line(centre_line, arrow_centre_line)
    if arrow_centre_line[0] >= 0 and arrow_centre_line[1] <= 0:
        angle_360 = angle_180

    if arrow_centre_line[0] >= 0 and arrow_centre_line[1] >= 0:
        angle_360 = angle_180
        

    if arrow_centre_line[0] <= 0 and arrow_centre_line[1] >= 0:
        angle_360 = 360 - angle_180

    if arrow_centre_line[0] <= 0 and arrow_centre_line[1] <= 0:
        angle_360 = 360 - angle_180
   
    return angle_360

# 打开摄像头
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

while cap.isOpened():
    retval, frame = cap.read()
    if not retval:
        print("无法读取帧")
        break
    
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 定义红色的HSV范围
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # 创建红色掩膜
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    # 对原图像和掩膜进行位运算
    red_regions = cv2.bitwise_and(frame, frame, mask=mask)

    # 转换为灰度图像
    gray = cv2.cvtColor(red_regions, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', gray)

    # 使用高斯模糊平滑图像
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 应用边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历所有轮廓
    for contour in contours:

        # 计算轮廓的周长
        perimeter = cv2.arcLength(contour, True)
        
        # 多边形拟合
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        # print(approx)
        # 如果多边形有四个顶点，则认为它是一个四边形
        
        if len(approx) == 4:
            
           
            # 获取四个顶点坐标
            points = approx.reshape(4, 2)
            # print(points)
            
            # 计算面积和宽高比
            area = cv2.contourArea(approx)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            
            # 过滤条件：面积在合理范围内，宽高比大致符合飞镖的比例
            if 1000 < area < 100000 and 1.5 < aspect_ratio < 2.5:
                
                # 寻找几何中心
                centre = find_centroid(approx, frame)
                centre = np.array(centre)
                
                # 画出轮廓（用于可视化）
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                
                print("四个顶点坐标：")
                for point in points:
                    print(tuple(point))
                    # 在图像上画出顶点（用于可视化）
                    cv2.circle(frame, tuple(point), 5, (0, 0, 255), -1)
                
                # 寻找顶点
                vertex = find_vertex(points)
                vertex = np.array(vertex)
                
                cv2.circle(frame, tuple(vertex), 5, (0, 255, 0), -1)
                # 假设frame是当前帧，width和height是帧的宽度和高度
                width = frame.shape[1]
                height = frame.shape[0]
                
                # 确定摄像头中心点
                high_center = np.array([width // 2, int(0)])
                bottom_center = np.array([width // 2, height])
                right_center = np.array([0, height // 2])
                left_center = np.array([width, height // 2])
                # frame_center = np.array([width // 2, height // 2])

                # 得到向量
                centre_line = high_center - bottom_center
                arrow_centre_line = vertex - centre
                angle_360 = get_angle(centre_line, arrow_centre_line)
                
            
                # 绘制摄像头中心线
                cv2.line(frame, high_center, bottom_center, (255, 255, 255), 1)
                cv2.line(frame, right_center, left_center, (255, 255, 255), 1)

                # 绘制检测到的线
                cv2.line(frame, vertex, centre, (255, 0, 0), 2)
               
                # 在图像上添加角度值
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f"Angle: {angle_360:.2f}°"
                cv2.putText(frame, text, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                print(f"线与摄像头中心线的夹角为：{angle_360}度")
       
        
    
    # 显示处理后的帧
    cv2.imshow('Live', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q退出
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
