import cv2
import numpy as np

# 全局变量，用于存储前几帧的质心位置
centroids = []
alpha = 0.9  # 滤波因子，控制新值和历史值的权重
last_cx , last_cy, last_angel = 0, 0, 0
last_vertex = (0, 0)
test = []
last_height = 0
flag_blue = 1 

def find_template_blue():
    template_image = cv2.imread('/home/lmz/opencv/blue_template.jpg')

    # 将图像从BGR转换到HSV色彩空间
    template_image_hsv = cv2.cvtColor(template_image, cv2.COLOR_BGR2HSV)    

    # 定义蓝色的HSV范围
    lower_blue = np.array([84, 179, 104])
    higher_blue = np.array([179, 255, 255])

    # 创建蓝掩膜
    mask = cv2.inRange(template_image_hsv, lower_blue, higher_blue)
    
    # 对原图像和掩膜进行位运算
    blue_regions = cv2.bitwise_and(frame, frame, mask=mask)

    # 转换为灰度图像
    gray = cv2.cvtColor(blue_regions, cv2.COLOR_BGR2GRAY)
    
    # 使用高斯模糊平滑图像
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 形态学闭操作
    new_image = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # cv2.imshow('new_image', new_image)

    # 应用边缘检测
    template_edges = cv2.Canny(new_image, 50, 150)

    # 寻找轮廓
    template_contours, _ = cv2.findContours(template_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    for contour in template_contours:

        # 计算轮廓的周长
        perimeter = cv2.arcLength(contour, True)
        
        # 多边形拟合
        approx = cv2.approxPolyDP(contour, 0.015 * perimeter, True)
        
        
        cv2.drawContours(template_image, contour, -1, (255, 0, 0), 2)
        cv2.drawContours(template_image, [approx], -1, (0, 255, 0), 2)
        # 如果多边形有四个顶点，则认为它是一个四边形

        if len(approx) == 4:
            
            return contour
            # 画出轮廓（用于可视化)

def find_template_red():
    template_image = cv2.imread('/home/lmz/opencv/red_template.jpg')

    # 将图像从BGR转换到HSV色彩空间
    template_image_hsv = cv2.cvtColor(template_image, cv2.COLOR_BGR2HSV)    

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

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

    # cv2.imshow('new_image', new_image)

    # 应用边缘检测
    template_edges = cv2.Canny(new_image, 50, 150)

    # 寻找轮廓
    template_contours, _ = cv2.findContours(template_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    for contour in template_contours:

        # 计算轮廓的周长
        perimeter = cv2.arcLength(contour, True)
        
        # 多边形拟合
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        cv2.drawContours(template_image, contour, -1, (255, 0, 0), 2)
        cv2.drawContours(template_image, [approx], -1, (0, 255, 0), 2)
        # print(approx)
        # 如果多边形有四个顶点，则认为它是一个四边形
        
        if len(approx) == 4:
           
            # print(points)
            return contour
           

def Lowpass(alpha, now, last):
    return alpha * now + (1 - alpha) * last

def find_centroid(contour, f, max=6):

    global centroids, alpha, last_cx, last_cy
    
    M = cv2.moments(contour)
    # print(f"矩: {M}")

    if M["m00"] != 0:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
    else:
        cX, cY = 0, 0
        print("无法计算质心")
    
    # 均值滤波
    centroids.append((cX, cY))
    if len(centroids) > max:
        centroids.pop(0)
  
    
    mean_cX = int(sum([cX for cX, cY in centroids]) / len(centroids))
    mean_cY = int(sum([cY for cX, cY in centroids]) / len(centroids))

    last_cx, last_cy = cX, cY
    
    # if abs(cX - last_cx) > 50 or abs(cY - last_cy) > 50:
    #     cX, cY = last_cx, last_cy
    # print("质心偏差过大，使用上一帧的质心")

    
    last_cx, last_cy = mean_cX, mean_cY
    
    # if abs(cX - last_cx) > 20 or abs(cY - last_cy) > 20:
    #     mean_cX, mean_cY = last_cx, last_cy
    # print("质心偏差过大，使用上一帧的质心")

    cv2.circle(f, (mean_cX, mean_cY), 5, (0, 0, 255), -1)
    
    print(f"质心坐标为：({mean_cX}, {mean_cY})")
    return (mean_cX, mean_cY)

def get_height_red(perceived_area, focal_length=612.62, known_area=15504):

    global last_height
    if perceived_area == 0:
        height = last_height
    else:
        height = focal_length * np.sqrt(known_area / perceived_area)
        last_height = height
    
    height = Lowpass(0.8, height, last_height)
    return height

def get_height_blue(perceived_area, focal_length=612.62, known_area=841.5):
    global last_height
    if perceived_area == 0:
        height = last_height
    else:
        height = focal_length * np.sqrt(known_area / perceived_area)
        last_height = height
    
    height = Lowpass(0.8, height, last_height)
    return height
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
        # 判断顶点与相邻的俩个点的距离
        if k == 0:
            angle = calculate_angle_point(points[k - 1], points[k + 1], vertex)
        else:
            angle = calculate_angle_point(points[(k + 1) % 4], points[k - 1], vertex)
        test.append(angle)
            
        if 80 <= angle <= 100:
            print(f"角度判据为{angle}")
            last_vertex = vertex
            return vertex 
    
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
# cap.set(cv2.CAP_PROP_GAIN, 1)  # 开启自动增益
# cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # 开启自动白平衡
cap.set(cv2.CAP_PROP_GAIN, 0)  # 关闭自动增益
cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # 关闭自动白平衡
cap.set(cv2.CAP_PROP_FPS, 120)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 


while cap.isOpened():
    retval, frame = cap.read()
    if not retval:
        print("无法读取帧")
        break
    
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 定义红色的HSV范围
    lower_red1 = np.array([0, 156, 67])
    upper_red1 = np.array([179, 255, 255])
    lower_red2 = np.array([164, 128, 64])
    upper_red2 = np.array([179, 255, 255])

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

    # 形态学闭操作
    new_frame = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # cv2.imshow('new_image', new_frame)

    # 应用边缘检测
    edges = cv2.Canny(new_frame, 50, 150)

    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历所有轮廓
    for contour in contours:

        # 计算轮廓的周长
        perimeter = cv2.arcLength(contour, True)
        
        # 多边形拟合
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        # print(approx)
        # 如果多边形有四个顶点，则认为它是一个四边形
        
        if len(approx) == 4:
            
            # 获取四个顶点坐标
            points = approx.reshape(4, 2)
            # print(points)
            
            # 计算面积和宽高比
            area = cv2.contourArea(approx)

            # 得到模板轮廓
            template_contour = find_template_red()

            # 匹配模板轮廓
            similarity = cv2.matchShapes(contour, template_contour, 1, 0.0)
            print(f"面积：{area}，匹配度：{similarity}")
            
            # 过滤条件：面积在合理范围内，    大致符合飞镖的比例
            if 2300 < area < 50000 and similarity < 0.5:

                flag_blue = 0
                
                # 获取高度
                height = get_height_red(area)
                cv2.putText(frame, f"height: {height:.2f}mm", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                # print(f"高度：{height}")
                
                # 寻找几何中心
                centre = find_centroid(contour, frame)
                
                # 在帧上显示质心坐标
                cv2.putText(frame, f"Centroid: ({centre[0]}, {centre[1]})", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
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
                last_angel = angle_360
                pass_angel = Lowpass(alpha, angle_360, last_angel)
            
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

    if flag_blue == 1:

        # 定义蓝色的HSV范围
        lower_blue = np.array([84, 179, 104])
        higher_blue = np.array([179, 255, 255])

        # 创建蓝掩膜
        mask = cv2.inRange(hsv, lower_blue, higher_blue)
        
        # 对原图像和掩膜进行位运算
        blue_regions = cv2.bitwise_and(frame, frame, mask=mask)

        # 转换为灰度图像
        gray = cv2.cvtColor(blue_regions, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Gray', gray)

        # 使用高斯模糊平滑图像
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 形态学闭操作
        new_frame = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        # cv2.imshow('new_image', new_frame)

        # 应用边缘检测
        edges = cv2.Canny(new_frame, 50, 150)

        # 寻找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 遍历所有轮廓
        for contour in contours:

            # 计算轮廓的周长
            perimeter = cv2.arcLength(contour, True)
            
            # 多边形拟合
            approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
            # print(approx)
            # 如果多边形有四个顶点，则认为它是一个四边形
            
            if len(approx) == 4:
                
                # 获取四个顶点坐标
                points = approx.reshape(4, 2)
                # print(points)
                
                # 计算面积和宽高比
                area = cv2.contourArea(contour)

                # 得到模板轮廓
                template_contour = find_template_blue()

                # 匹配模板轮廓
                similarity = cv2.matchShapes(contour, template_contour, 1, 0.0)
                print(f"面积：{area}，匹配度：{similarity}")
                
                # 过滤条件：面积在合理范围内，    大致符合飞镖的比例
                if 2300 < area < 50000 and similarity < 0.5:
                    
                    # 获取高度
                    height = get_height_blue(area)
                    cv2.putText(frame, f"height: {height:.2f}mm", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    # print(f"高度：{height}")
                    
                    # 寻找几何中心
                    centre = find_centroid(contour, frame)
                    centre = np.array(centre) 
                    
                    # 画出轮廓（用于可视化）
                    cv2.drawContours(frame, [approx], -1, (255, 255, 255), 2) ### 白色轮廓
                    
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
                    last_angel = angle_360
                    pass_angel = Lowpass(alpha, angle_360, last_angel)
                
                    # 绘制摄像头中心线
                    cv2.line(frame, high_center, bottom_center, (255, 255, 255), 1)
                    cv2.line(frame, right_center, left_center, (255, 255, 255), 1)

                    # 绘制检测到的线
                    cv2.line(frame, vertex, centre, (255, 0, 0), 2)
                
                    # 在图像上添加角度值
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text = f"Angle: {angle_360:.2f}°"
                    cv2.putText(frame, text, (550, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA) ## 右上角字体
                    print(f"线与摄像头中心线的夹角为：{angle_360}度")

    # 显示处理后的帧
    cv2.imshow('Live', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q退出
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
