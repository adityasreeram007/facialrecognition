import cv2
import numpy as numpy

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
print(rec.read('trainingData.yml'))
id = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5);
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
        id, conf = rec.predict(gray[y:y+h, x:x+w])
        if id==1:
            cv2.putText(img,"Name-karthi",(x,y+h), font, 1,(0,0,0),2,cv2.LINE_AA)
            cv2.putText(img,"Age-20",(x,y+(20+h)), font, 1,(0,0,0),2,cv2.LINE_AA)
            cv2.putText(img,"Sex-male",(x,y+(40+h)), font, 1,(0,0,0),2,cv2.LINE_AA)
        elif id ==2:
            cv2.putText(img,"Name-devi",(x,y+h), font, 1,(0,0,0),2,cv2.LINE_AA)
            cv2.putText(img,"Age-20",(x,y+(20+h)), font, 1,(0,0,0),2,cv2.LINE_AA)
            cv2.putText(img,"Sex-Female",(x,y+(40+h)), font, 1,(0,0,0),2,cv2.LINE_AA)
        else:
            cv2.putText(img,"NON AUTHORIZED",(x,y+h), font, 2,(255,255,255),2,cv2.LINE_AA)
        pass
    cv2.imshow("Faces",img)
    if(cv2.waitKey(1) == ord('q')):
        break
    pass
cam.release()
cv2.destroyAllWindows()
