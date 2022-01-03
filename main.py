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
        
        results =  face_recognition.face_distance([userEncodes], encode)
        result = numpy.average(results)

        if(result < 0.05):
            return user
    return "Unknown"


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faceLogs = face_recognition.face_locations(rgb)
    encodeLogs = face_recognition.face_encodings(rgb)

    if(faceLogs != [] and encodeLogs != []):
        faceLog = faceLogs[0]
        encodeLog = encodeLogs[0]
        cv2.rectangle(img, (faceLog[3], faceLog[0]), (faceLog[1], faceLog[2]), (0, 0, 255), 2)
        result = FindResult(encodeLog)
        cv2.putText(img, result, (faceLog[3], faceLog[0]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow("Video", img)

cap.release()
cv2.destroyAllWindows()