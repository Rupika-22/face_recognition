import cv2
from deepface import DeepFace
import numpy as np
from scipy.spatial.distance import cosine
import os
os.environ["DEEPFACE_HOME"] = r"E:\New folder\file\deepface"
known=[]
name=[]
f=r"E:\New folder\demo"
for i in os.listdir(f):
    n=os.path.join(f,i)
    name.append(os.path.splitext(i)[0])
    f1=cv2.imread(n)
    d=DeepFace.represent(f1,model_name="ArcFace",detector_backend="opencv",enforce_detection=False)[0]["embedding"]
    n=np.array(d).flatten()
    known.append(n)
v=cv2.VideoCapture(0)
while True:
    _,frame=v.read()
    em=DeepFace.represent(frame,model_name="ArcFace",detector_backend="opencv",enforce_detection=False)[0]["embedding"]
    d1=np.array(em).flatten()
    name1="unknown"
    for index,k in enumerate(known):
        c1=cosine(d1,k)
        if c1<=0.6:
            name1=name[index]
            break
    c=cv2.CascadeClassifier(r"E:\New folder\algorithm.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = c.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 0), 2)
        cv2.putText(frame, name1, (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    image=cv2.imshow("video",frame)
    if cv2.waitKey(1)==ord("q"):
        break
v.release()
cv2.destroyAllWindows()
