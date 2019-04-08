import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os

#   Face Detect
###############################################################################
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
datapath = "./data/"
ans = True
while ans == True:
    filename = input("Enter Face ID : ")
    face_data = []
    cnt = 0
    for i in range(401):
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
            #cv2.imshow('Extracted frame',face_section)
            if cnt%10 == 0:
                print(cnt)
                face_data.append(face_section)
        cnt += 1
        cv2.imshow('Video Frame',frame)
        cv2.waitKey(1)
        #Show frame
    #end loop

    face_data = np.asarray(face_data)
    face_data = face_data.reshape((face_data.shape[0],-1))
    np.save(datapath + filename + '.npy',face_data)
    print('Face Data saved at : ' + datapath + filename + '.npy')
    ans = input("Do you want to enter new data(True/False)? ")
#end loop

#KNN Classifier
################################################################################
classifier = KNeighborsClassifier()


#Training
################################################################################
class_id = 0
name = {}
face_data = []
label_data = []

for fx in os.listdir(datapath):
    if fx.endswith('.npy'):
        name[class_id] = fx[:-4]
        data_item = np.load(datapath+fx)
        face_data.append(data_item)

        target = class_id*np.ones((data_item.shape[0],1))
        label_data.append(target)
face_data = np.concatenate(face_data,axis = 0)
label_data = np.concatenate(label_data,axis = 0).reshape((-1,1))

print(face_data.shape)
print(label_data.shape)

#Testing
################################################################################

classifier.fit(face_data,label_data)

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
        #Extract faces
        offset = 10
        face_section = gray_frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        output = classifier.predict([face_section.flatten()])
        cv2.putText(frame,name[output[0]],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    #Show frame
    cv2.imshow('Video Frame',frame)
    key = cv2.waitKey(1) & 0xFF
    if(key == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
