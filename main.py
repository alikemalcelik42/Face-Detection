import cv2
import numpy
import face_recognition
import os

data = {}

def GetData():
    for folder in os.listdir("imgs"):
        data[folder] = []
        for file in os.listdir(f"imgs/{folder}"):
            img = face_recognition.load_image_file(f"imgs/{folder}/{file}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faceLog = face_recognition.face_locations(img)[0]
            encode = face_recognition.face_encodings(img)[0]

            data[folder].append({
                "img": img,
                "faceLog": faceLog,
                "encode": encode
            })

GetData()


def FindResult(encode):
    for user in data.keys():
        userEncodes = []
        for img in data[user]:
            userEncodes.append(img["encode"])
        
        result =  face_recognition.compare_faces(userEncodes, encode)

        if(result.count(True) > result.count(False)):
            return user
    return "Unknown"


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faceLogs = face_recognition.face_locations(rgb)
    encodeLogs = face_recognition.face_encodings(rgb)

    i = 0

    while i < len(faceLogs): 
        faceLog = faceLogs[i]
        encodeLog = encodeLogs[i]
        result = FindResult(encodeLog)

        cv2.rectangle(img, (faceLog[3], faceLog[0]), (faceLog[1], faceLog[2]), (0, 255, 0), 2)
        cv2.rectangle(img, (faceLog[3], faceLog[2]+35), (faceLog[1], faceLog[2]), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, result, (faceLog[3]+6, faceLog[2]+27), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        
        i += 1

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    cv2.imshow("Video", img)

cap.release()
cv2.destroyAllWindows()