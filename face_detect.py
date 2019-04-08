import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
datapath = "./data/"
filename = input("Enter Face ID : ")
face_data = []
cnt = 0
while True:
    ret,frame = cap.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #If frame is not captured correctly skip frame
    if ret == False:
        continue
    #detectMultiscale returns a list of tuples of form (x,y,w,h)
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
    faces = sorted(faces,key = lambda t : t[2]*t[3], reverse = True)
    #draw rectangle
    for face in faces:
        (x,y,w,h) = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        #Extract faces
        offset = 10
        face_section = gray_frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        cv2.imshow('Extracted frame',face_section)

        if cnt%10 == 0:
            print(cnt)
            face_data.append(face_section)
    cnt += 1

    #Show frame
    cv2.imshow('Video Frame',frame)
    #end loop on q press
    key = cv2.waitKey(1) & 0xFF
    if(key == ord('q')):
        break
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
np.save(datapath + filename + '.npy',face_data)
print('Face Data saved at : ' + datapath + filename + '.npy')
cap.release()
cv2.destroyAllWindows()
