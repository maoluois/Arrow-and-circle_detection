import cv2
import numpy as np
from math import sqrt, acos, pi, pow, atan2

camera_a = 469.2946
camera_b = 0.0075

def calc_real_length(L, height):
    return L * (height / camera_a + camera_b)

def calc_height_red_signS(S):
    L = sqrt(S / 0.1967) / 45
    return camera_a / (L - camera_b)

def calc_height_blue_signS(S):
    L = sqrt(S / 0.1776666666) / 15
    return camera_a / (L - camera_b)

def fit_line(pts):
    sum_x = sum_y = sum_x2 = sum_y2 = sum_xy = 0
    n = len(pts)

    for pt in pts:
        sum_x += pt[0]
        sum_y += pt[1]
        sum_x2 += pt[0] * pt[0]
        sum_y2 += pt[1] * pt[1]
        sum_xy += pt[0] * pt[1]

    namda = sum_x2 * n - sum_x * sum_x
    if namda < 0.001:
        return 1000000
    namda = 1.0 / namda

    A = (sum_xy * n - sum_x * sum_y) * namda
    B = (sum_x2 * sum_y - sum_x * sum_xy) * namda
    return (A * A * sum_x2 + sum_y2 + n * B * B + 2 * A * B * sum_x - 2 * A * sum_xy - 2 * B * sum_y) / n

def angle(pt1, pt2, pt0):
    dx1 = pt1[0] - pt0[0]
    dy1 = pt1[1] - pt0[1]
    dx2 = pt2[0] - pt0[0]
    dy2 = pt2[1] - pt0[1]
    sin = dx1 * dy2 - dy1 * dx2
    if sin > 0:
        return acos((dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10))
    else:
        return -acos((dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10))

def balance_white(src):
    imageRGB = cv2.split(src)

    B = np.mean(imageRGB[0])
    G = np.mean(imageRGB[1])
    R = np.mean(imageRGB[2])

    KB = (R + G + B) / (3 * B)
    KG = (R + G + B) / (3 * G)
    KR = (R + G + B) / (3 * R)

    imageRGB[0] = imageRGB[0] * KB
    imageRGB[1] = imageRGB[1] * KG
    imageRGB[2] = imageRGB[2] * KR

    src = cv2.merge(imageRGB)
    return src

def chk(frame, dev, approx_poly, corners, SL):
    dev = cv2.morphologyEx(dev, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    
    contoursR, _ = cv2.findContours(dev, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contoursR:
        S = cv2.contourArea(contour)
        if S < 200:
            continue

        L = cv2.arcLength(contour, True)
        e = L * L / 56.49 / S
        if e < 0.3 or e > 1.3:
            continue

        approx = cv2.approxPolyDP(contour, L * 0.05, True)
        if len(approx) < 4 or len(approx) > 6:
            continue

        approx2 = []
        last_point = None
        sum_x = sum_y = 0
        n = 0
        min_distance = L * 0.001
        min_distance *= min_distance

        for i in range(len(approx)):
            x = approx[i][0][0] - approx[i - 1][0][0]
            y = approx[i][0][1] - approx[i - 1][0][1]
            if x * x + y * y > min_distance:
                if n > 0:
                    approx2.append([int(sum_x / n), int(sum_y / n)])
                    sum_x = sum_y = n = 0
                approx2.append(approx[i][0])
            else:
                if n == 0:
                    last_point = approx[i][0]
                sum_x += x
                sum_y += y
                n += 1

        if n > 0:
            approx2.append([int(sum_x / n), int(sum_y / n)])

        if len(approx2) != 4:
            continue

        current_corners = [0, 0, 0, 0]
        sq = 0
        sign = 0

        for j in range(2, 6):
            p1 = j % 4
            p2 = (j - 1) % 4

            current_angle = angle(approx2[p1], approx2[j - 2], approx2[p2]) * 180 / pi
            abs_angle = abs(current_angle)

            if abs_angle > 70:
                if current_corners[0] != 0:
                    break
                else:
                    if sign == 0:
                        sign = 1 if current_angle > 0 else -1
                    elif sign * current_angle < 0:
                        break
                    sq += pow(abs_angle - 83.9, 2)
                    current_corners[0] = p2 + 1
            elif abs_angle > 35:
                if current_corners[2] != 0:
                    break
                else:
                    if sign == 0:
                        sign = 1 if current_angle < 0 else -1
                    elif sign * current_angle > 0:
                        break
                    sq += pow(abs_angle - 54.1, 2)
                    current_corners[2] = p2 + 1
            else:
                if sign == 0:
                    sign = 1 if current_angle < 0 else -1
                elif sign * current_angle > 0:
                    break
                sq += pow(abs_angle - 18.1, 2)
                if current_corners[1] != 0:
                    if current_corners[3] != 0:
                        break
                    else:
                        current_corners[3] = p2 + 1
                else:
                    current_corners[1] = p2 + 1

        sq /= 4
        if sq > 150:
            continue

        for j in range(1, 4):
            e = abs(current_corners[j] - current_corners[j - 1])
            if e != 1 and e != 3:
                break

        p1_x = approx2[current_corners[1] - 1][0]
        p1_y = approx2[current_corners[1] - 1][1]
        p3_x = approx2[current_corners[3] - 1][0]
        p3_y = approx2[current_corners[3] - 1][1]
        p2_x = approx2[current_corners[2] - 1][0]
        p2_y = approx2[current_corners[2] - 1][1]

        p21_x = p1_x - p2_x
        p21_y = p1_y - p2_y
        p23_x = p3_x - p2_x
        p23_y = p3_y - p2_y

        if p21_x * p23_y - p21_y * p23_x > 0:
            approx2[current_corners[1] - 1] = [p3_x, p3_y]
            approx2[current_corners[3] - 1] = [p1_x, p1_y]

        approx_poly.append(approx2)
        corners.append(current_corners)
        SL.append([S, L])
        cv2.polylines(frame, [np.array(approx2)], True, (255, 0, 255), 2)

    return len(approx_poly) > 0

def match(frame, template_R, template_B, approx_poly, corners, SL):
    matched_index = []

    while len(matched_index) < len(approx_poly):
        max_S = 0
        selected_poly = -1

        for j in range(len(approx_poly)):
            if j in matched_index:
                continue
            if SL[j][0] > max_S:
                selected_poly = j

        if selected_poly == -1:
            break

        p0 = approx_poly[selected_poly][corners[selected_poly][0] - 1]
        p1 = approx_poly[selected_poly][corners[selected_poly][1] - 1]
        p2 = approx_poly[selected_poly][corners[selected_poly][2] - 1]
        p3 = approx_poly[selected_poly][corners[selected_poly][3] - 1]

        w = sqrt((p0[0] - p3[0]) ** 2 + (p0[1] - p3[1]) ** 2)
        h = sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        if h > w:
            w, h = h, w

        if h > w * 2 or h * 1.5 < w:
            matched_index.append(selected_poly)
            continue

        p0_x = p0[0] + (p1[0] - p0[0]) * 0.18
        p0_y = p0[1] + (p1[1] - p0[1]) * 0.18
        p2_x = p2[0] + (p3[0] - p2[0]) * 0.18
        p2_y = p2[1] + (p3[1] - p2[1]) * 0.18
        p0 = [int(p0_x), int(p0_y)]
        p2 = [int(p2_x), int(p2_y)]

        if len(p0) != 2 or len(p2) != 2:
            continue

        p0_p2_distance = sqrt((p0[0] - p2[0]) ** 2 + (p0[1] - p2[1]) ** 2)

        R = atan2(p2[0] - p0[0], p2[1] - p0[1]) * 180 / pi
        if R < 0:
            R += 180
        if R > 45 and R < 135:
            continue

        template_poly = [p0, p1, p2, p3]

        for i in range(len(template_R)):
            for j in range(len(template_R[i])):
                p = template_R[i][j]
                d = sqrt((p0[0] - p[0]) ** 2 + (p0[1] - p[1]) ** 2)
                if d < p0_p2_distance * 0.2:
                    break
            else:
                template_R[i].append(p0)
                template_R[i].append(p2)
                break
        else:
            template_R.append([p0, p2])

        for i in range(len(template_B)):
            for j in range(len(template_B[i])):
                p = template_B[i][j]
                d = sqrt((p0[0] - p[0]) ** 2 + (p0[1] - p[1]) ** 2)
                if d < p0_p2_distance * 0.2:
                    break
            else:
                template_B[i].append(p0)
                template_B[i].append(p2)
                break
        else:
            template_B.append([p0, p2])

        matched_index.append(selected_poly)

    return len(template_R) > 0 or len(template_B) > 0

# Usage example with frame and other necessary inputs
# frame = cv2.imread('path_to_image')
# dev = cv2.imread('path_to_image', 0)
# approx_poly = []
# corners = []
# SL = []

# result = chk(frame, dev, approx_poly, corners, SL)
# print(result)


def findRectandTri(frame, squares, triangles, SquareCenters, TriangleCenters):
    # Your implementation for detecting rectangles and triangles
    # This is a placeholder function
    pass

def draw(frame, shapes):
    for shape in shapes:
        cv2.drawContours(frame, [np.array(shape)], -1, (0, 255, 0), 2)

def getCenterPoint(SquareCenters, TriangleCenters, aera_of_square, squares):
    if squares:
        center = np.mean(squares[0], axis=0)
        aera_of_square = cv2.contourArea(np.array(squares[0]))
    else:
        center = (0, 0)
        aera_of_square = 0
    return center

def help():
    print("\nA program using pyramid scaling, Canny, contours, contour simplification and\n"
          "memory storage (it's got it all folks) to find\n"
          "squares in a list of images pic1-6.png\n"
          "Returns sequence of squares detected on the image.\n"
          "the sequence is stored in the specified memory storage\n"
          "Call:\n./squares\nUsing OpenCV version {}\n".format(cv2.__version__))

def angle(pt1, pt2, pt0):
    dx1 = pt1[0] - pt0[0]
    dy1 = pt1[1] - pt0[1]
    dx2 = pt2[0] - pt0[0]
    dy2 = pt2[1] - pt0[1]
    return (dx1 * dx2 + dy1 * dy2) / np.sqrt((dx1*dx1 + dy1*dy1) * (dx2*dx2 + dy2*dy2) + 1e-10)

def find_squares(image):
    squares = []
    squares_centers = []
    blurred = cv2.GaussianBlur(image, (3, 3), 3)
    gray0 = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    gray = np.zeros_like(gray0)
    for c in range(3):
        channel = blurred[:, :, c]
        for l in range(5):
            if l == 0:
                gray = cv2.Canny(channel, 0, 50, apertureSize=5)
                gray = cv2.dilate(gray, None)
            else:
                _, gray = cv2.threshold(channel, (l + 1) * 255 // 5, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour_len = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, contour_len * 0.02, True)
                if len(approx) == 4 and cv2.contourArea(approx) > 1000 and cv2.isContourConvex(approx):
                    max_cosine = 0
                    for j in range(2, 5):
                        cosine = abs(angle(approx[j % 4][0], approx[j - 2][0], approx[j - 1][0]))
                        max_cosine = max(max_cosine, cosine)
                    if max_cosine < 0.3:
                        squares.append(approx)
                        squares_centers.append(get_center(approx))
    return squares, squares_centers

def get_center(points):
    center = np.mean(points, axis=0)
    return tuple(center[0])

def find_rect_and_tri(image):
    squares = []
    triangles = []
    squares_centers = []
    triangle_centers = []
    blurred = cv2.GaussianBlur(image, (3, 3), 3)
    gray0 = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    gray = np.zeros_like(gray0)
    for c in range(3):
        channel = blurred[:, :, c]
        for l in range(5):
            if l == 0:
                gray = cv2.Canny(channel, 0, 50, apertureSize=5)
                gray = cv2.dilate(gray, None)
            else:
                _, gray = cv2.threshold(channel, (l + 1) * 255 // 5, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour_len = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, contour_len * 0.02, True)
                if len(approx) == 4 and cv2.contourArea(approx) > 1000 and cv2.isContourConvex(approx):
                    max_cosine = 0
                    for j in range(2, 5):
                        cosine = abs(angle(approx[j % 4][0], approx[j - 2][0], approx[j - 1][0]))
                        max_cosine = max(max_cosine, cosine)
                    if max_cosine < 0.3:
                        squares.append(approx)
                        squares_centers.append(get_center(approx))
                if len(approx) == 3 and cv2.contourArea(approx) > 1000 and cv2.isContourConvex(approx):
                    triangles.append(approx)
                    triangle_centers.append(get_center(approx))
    return squares, triangles, squares_centers, triangle_centers

def draw(image, contours):
    for contour in contours:
        cv2.polylines(image, [contour], True, (0, 255, 0), 3, cv2.LINE_AA)

def draw_both(image, triangles, squares):
    draw(image, triangles)
    draw(image, squares)
    
def main():
    frame = cv2.imread("./images_2/image_428.bmp")
    
    squares = []
    triangles = []
    SquareCenters = []
    TriangleCenters = []
    aera_of_square = 0.0
    
    findRectandTri(frame, squares, triangles, SquareCenters, TriangleCenters)
    draw(frame, squares)
    draw(frame, triangles)
    
    center = getCenterPoint(SquareCenters, TriangleCenters, aera_of_square, squares)
    print(f"The position of center is ({center[0]}, {center[1]})")
    
    cv2.imshow("Square Detection Demo", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
