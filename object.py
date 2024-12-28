import numpy as np
import imutils
import cv2
import time

prototxt="deploy.prototxt"
model="mobilenet_iter_73000.caffemodel"
confThresh=0.2   #confident about threshhold value
CLASSES=["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","Lvmonitor","mobile"]
COLORS=np.random.uniform(0,255,size=(len(CLASSES),3)) #generate random color in RGB 3->RGB
print("Loading model...")
net=cv2.dnn.readNetFromCaffe(prototxt,model)
print("Model Loaded")
print("Starting Camera Feed....")
vs=cv2.VideoCapture(0)  #camera id initialization
time.sleep(2.0)
while True:
    _,frame=vs.read()
    frame=imutils.resize(frame,width=1000)
    (h,w)=frame.shape[:2]  #for height,width
    imResizeBlob=cv2.resize(frame,(300,300))  #for image frame
    blob=cv2.dnn.blobFromImage(imResizeBlob,0.007843,(300,300),127.5)
    net.setInput(blob)  #input is given to caffee model
    detections=net.forward()  #for getting id,confident level,coordinates
    detShape=detections.shape[2]
    for i in np.arange(0,detShape):
        confidence=detections[0,0,i,2]  #2=>confidence level
        if confidence>confThresh:
            idx = int(detections[0,0,i,1]) #1=>id
            print("ClassID:",detections[0,0,i,1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            box = box.astype("int")  # Convert box coordinates to integers
            # Unpack the box coordinates
            (startX, startY, endX, endY) = box

            label="{}:{:.2f}%".format(CLASSES[idx],confidence*100)
            cv2.rectangle(frame,(startX,startY),(endX,endY),COLORS[idx],2)
            if startY-15>15:
                y=startY-15
            else:
                startY+15
            cv2.putText(frame,label,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[idx],2)
    cv2.imshow("Frame",frame)
    key=cv2.waitKey(1)
    if key==27:
        break
vs.release()
cv2.destroyAllWindows()
            

