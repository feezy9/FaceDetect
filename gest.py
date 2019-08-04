import cv2

frontalPalm_cascade = cv2.CascadeClassifier(r"C:\Users\surface\Projects\dev\temp\pytemp\FaceDetect\Opencv\haarcascade\closed_frontal_palm.xml")
fist_cascade = cv2.CascadeClassifier(r"C:\Users\surface\Projects\dev\temp\pytemp\FaceDetect\Opencv\haarcascade\fist.xml")
palm_cascade = cv2.CascadeClassifier(r"C:\Users\surface\Projects\dev\temp\pytemp\FaceDetect\Opencv\haarcascade\palm.xml")
gest_cascade = cv2.CascadeClassifier(r"C:\Users\surface\Projects\dev\temp\pytemp\FaceDetect\Opencv\haarcascade\aGest.xml")
faceCascade = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")     
cap = cv2.VideoCapture(0)

while cap.isOpened():
   ret, frame = cap.read()
   if ret:
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     
     faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
     )

     for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                print(x,y)

     fists = fist_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(50, 50)
                )
     for (x, y, w, h) in fists:
               cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
               roi_gray = gray[y:y + h, x:x + w]
               roi_color = frame[y:y + h, x:x + w]
     palms = palm_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(50, 50)
                )
     for (px, py, pw, ph) in palms:
               cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 255, 0), 2)
               roi_gray = gray[py:py + ph, px:px + pw]
               roi_color = frame[py:py + ph, px:px + pw]
     frontFists = frontalPalm_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(50, 50)
                )
     for (ffx, ffy, ffw, ffh) in frontFists:
               cv2.rectangle(frame, (ffx, ffy), (ffx + ffw, ffy + ffh), (0, 255, 0), 2)
               roi_gray = gray[ffy:ffy + ffh, ffx:ffx + ffw]
               roi_color = frame[ffy:ffy + ffh, ffx:ffx + ffw]
     gestures = gest_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(50, 50)
                )
     for (gx, gy, gw, gh) in gestures:
               cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
               roi_gray = gray[gy:gy + gh, gx:gx + gw]
               roi_color = frame[gy:gy + gh, gx:gx + gw]   
   
   horizontal_img = cv2.flip( frame, 1 )
   cv2.imshow("hand found", horizontal_img)

   if cv2.waitKey(1) & 0xFF == ord('q'): #press q to exit program
         break

cv2.destroyAllWindows()
cap.release()   