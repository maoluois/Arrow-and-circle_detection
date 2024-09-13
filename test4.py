import cv2
import numpy as np
from math import sqrt, acos, pow

# Constants
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
        sum_x2 += pt[0] ** 2
        sum_y2 += pt[1] ** 2
        sum_xy += pt[0] * pt[1]
    
    namda = sum_x2 * n - sum_x ** 2
    if namda < 0.001:
        return 1000000
    namda = 1.0 / namda

    A = (sum_xy * n - sum_x * sum_y) * namda
    B = (sum_x2 * sum_y - sum_x * sum_xy) * namda
    return (A * A * sum_x2 + sum_y2 + n * B * B + 2 * A * B * sum_x - 2 * A * sum_xy - 2 * B * sum_y) / n, A, B

def angle(pt1, pt2, pt0):
    dx1 = pt1[0] - pt0[0]
    dy1 = pt1[1] - pt0[1]
    dx2 = pt2[0] - pt0[0]
    dy2 = pt2[1] - pt0[1]
    sin_val = dx1 * dy2 - dy1 * dx2
    if sin_val > 0:
        return acos((dx1 * dx2 + dy1 * dy2) / sqrt((dx1 ** 2 + dy1 ** 2) * (dx2 ** 2 + dy2 ** 2) + 1e-10))
    else:
        return -acos((dx1 * dx2 + dy1 * dy2) / sqrt((dx1 ** 2 + dy1 ** 2) * (dx2 ** 2 + dy2 ** 2) + 1e-10))

def chk(frame):
    approx_poly = []
    corners = []

    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    contoursR, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_size = len(contoursR)

    for i in range(contours_size):
        continue_flag = False

        S = cv2.contourArea(contoursR[i], False)
        if S < 200:
            continue

        L = cv2.arcLength(contoursR[i], True)
        e = L * L / 56.49 / S
        if e < 0.5 or e > 1.5:
            continue

        approx = cv2.approxPolyDP(contoursR[i], L * 0.05, True)
        if len(approx) < 4 or len(approx) > 10:
            continue

        approx2 = []
        last_point = None
        sum_x = sum_y = n = 0
        min_distance = L * 0.05
        min_distance *= min_distance

        for j in range(len(approx)):
            x = approx[j][0][0] - (approx[j-1][0][0] if j > 0 else approx[-1][0][0])
            y = approx[j][0][1] - (approx[j-1][0][1] if j > 0 else approx[-1][0][1])
            if x ** 2 + y ** 2 > min_distance:
                if n > 0:
                    approx2.append([sum_x / n, sum_y / n])
                    sum_x = sum_y = n = 0
                approx2.append(approx[j][0])
            else:
                if n > 0:
                    last_point = approx[j][0]
                sum_x += approx[j][0][0]
                sum_y += approx[j][0][1]
                n += 1
        if n > 0:
            approx2.append([sum_x / n, sum_y / n])
            sum_x = sum_y = n = 0

        if len(approx2) != 4:
            continue

        current_corners = [0, 0, 0, 0]
        sq = 0
        sign = 0

        for j in range(2, 6):
            p1 = j if j < 4 else j - 4
            p2 = j - 1 if j < 4 else j - 5

            current_angle = angle(approx2[p1], approx2[j - 2], approx2[p2]) * 180 / 3.1415926
            abs_angle = abs(current_angle)

            if abs_angle > 70:
                if current_corners[0] != 0:
                    continue_flag = True
                    break
                else:
                    if sign == 0:
                        sign = 1 if current_angle > 0 else -1
                    elif sign * current_angle < 0:
                        continue_flag = True
                        break
                    sq += pow(abs_angle - 83.9, 2)
                    current_corners[0] = p2 + 1
            elif abs_angle > 35:
                if current_corners[2] != 0:
                    continue_flag = True
                    break
                else:
                    if sign == 0:
                        sign = 1 if current_angle < 0 else -1
                    elif sign * current_angle > 0:
                        continue_flag = True
                        break
                    sq += pow(abs_angle - 54.1, 2)
                    current_corners[2] = p2 + 1
            else:
                if sign == 0:
                    sign = 1 if current_angle < 0 else -1
                elif sign * current_angle > 0:
                    continue_flag = True
                    break
                sq += pow(abs_angle - 18.1, 2)
                if current_corners[1] != 0:
                    if current_corners[3] != 0:
                        continue_flag = True
                        break
                    else:
                        current_corners[3] = p2 + 1
                else:
                    current_corners[1] = p2 + 1
        
        if continue_flag:
            continue

        sq /= 4
        for j in range(1, 4):
            e = abs(current_corners[j] - current_corners[j - 1])
            if e != 1 and e != 3:
                continue_flag = True
                break
        if continue_flag:
            continue
        if sq > 200:
            continue

        p1_x, p1_y = approx2[current_corners[1] - 1]
        p3_x, p3_y = approx2[current_corners[3] - 1]
        p2_x, p2_y = approx2[current_corners[2] - 1]
        p21_x, p21_y = p1_x - p2_x, p1_y - p2_y
        p23_x, p23_y = p3_x - p2_x, p3_y - p2_y
        if p21_x * p23_y - p21_y * p23_x > 0:
            approx2[current_corners[1] - 1] = [p3_x, p3_y]
            approx2[current_corners[3] - 1] = [p1_x, p1_y]

        corners.append(current_corners)
        approx_poly.append(approx2)

    return len(approx_poly) > 0, approx_poly, corners

def floodFillPostprocess(img, colorDiff=(1, 1, 1)):
    assert not img is None
    rng = np.random.default_rng()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if mask[y + 1, x + 1] == 0:
                newVal = rng.integers(256, size=3)
                cv2.floodFill(img, mask, (x, y), newVal.tolist(), colorDiff, colorDiff)

def meanShiftSegmentation(img):
    spatialRad = 10
    colorRad = 10
    maxPyrLevel = 1
    res = cv2.pyrMeanShiftFiltering(img, spatialRad, colorRad, maxPyrLevel)
    floodFillPostprocess(res, (2, 2, 2))
    cv2.imshow("flood", res)
    return res

def main():
    print("dd")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera")
        return
    
    print("ss")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame")
            break

        img = meanShiftSegmentation(frame)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        found, approx_poly, corners = chk(thresh)
        print(f"found = {found}")
        
        if found:
            for approx, current_corners in zip(approx_poly, corners):
                color = (0, 255, 255)
                cv2.drawContours(frame, [np.array(approx, dtype=np.int32)], 0, color, 2, cv2.LINE_AA)

                for j in range(4):
                    cv2.circle(frame, tuple(approx[current_corners[j] - 1]), 3, (0, 0, 255), 3, cv2.LINE_AA)

                length_sum = 0
                for j in range(4):
                    length_sum += sqrt((approx[j][0] - approx[j - 1][0]) ** 2 + (approx[j][1] - approx[j - 1][1]) ** 2)
                length_sum /= 4
                length = calc_real_length(length_sum, 1)
                text = f"length = {length:.2f}"
                cv2.putText(frame, text, (approx[0][0], approx[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Result", frame)
        
        if cv2.waitKey(30) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
