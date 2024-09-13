import cv2
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_GAIN, 0)  # 关闭自动增益

cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # 关闭自动白平衡

while(cap.isOpened()):
    retval, frame = cap.read()
    cv2.imshow('Live', frame)
  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()