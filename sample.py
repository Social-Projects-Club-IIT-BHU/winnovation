import cv2
import numpy as np

# define a video capture object 
cap = cv2.VideoCapture('/media/amshra267/sa44/winnovation/sample_videos/00414.mp4') 

i=0
print(cap.isOpened())
while(True):
    ret, frame = cap.read()
    print(ret)
    if ret == False:
        break
  #  cv2.imwrite('kang'+str(i)+'.jpg',frame)
    i+=1
    print(i)
 
cap.release()
cv2.destroyAllWindows()




