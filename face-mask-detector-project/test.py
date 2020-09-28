import cv2
import numpy as np
from keras.models import load_model
model=load_model("D:\\ML projects\\face-mask-detector-project\\models\\model2-003.h5")

# Nhận dạng tên
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('D:\\ML projects\\name_prediction\\dataset\\model\\trainer.yml')
#faceCascade = cv2.CascadeClassifier('C:\\Users\\nvtda\\anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

#iniciate id counter
idz = 0
# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'DatDepZai', 'TaylorSwift', 'Leonardo', 'ChauTinhTinh']


labels_dict={0: 'no mask',1:'mask'}
color_dict={0:(0,0,255),1:(0,255,0)}

size = 4
webcam = cv2.VideoCapture(0) #Use camera 0

# We load the xml file
classifier = cv2.CascadeClassifier('C:\\Users\\nvtda\\anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

while True:
    (rval, im) = webcam.read()
    im=cv2.flip(im,1,1) #Flip to act as a mirror

    # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    # detect MultiScale / faces 
    faces = classifier.detectMultiScale(mini)

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        #Save just the rectangle faces in SubRecFaces
        face_img = im[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(150,150))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        #print(result)
        
        label=np.argmax(result,axis=1)[0]
        
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        idz, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        # If confidence is less them 100 ==> "0" : perfect match 
        
        
        cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
        
        if (confidence < 100):
            idz = names[idz]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            idz = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        cv2.putText(im, str(idz), (x+w,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        #cv2.putText(im, labels_dict[label], (x+w,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
       
        cv2.putText(im, str(confidence), (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1)
        
    # Show the image
    cv2.imshow('LIVE',   im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop 
    if key == 27: #The Esc key
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()