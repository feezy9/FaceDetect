import cv2
import sys
import keyboard
import time

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )
    
    changeN = False
    changeP = False
      # if changeN:
  
    # def changeSong():
    #     keyboard.send(-176)

    fired = 0

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if x < 100:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,'Next',(0,200), font, 1, (200,255,155), 2, cv2.LINE_AA)
            while fired == 0:
                fired += 1
                keyboard.send(-176)
                time.sleep(1)
                break
        # elif x > 100:
        #     fired += 1
               
        # while x > 300: 
        #     if changeP == False:
        #         changeP = True
        #         break

    horizontal_img = cv2.flip( frame, 1 )
    # Display the resulting frame
    cv2.imshow('Video', horizontal_img)
  
       

    # if changeP:
    #     keyboard.send(-177)
    print(changeN, fired)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()