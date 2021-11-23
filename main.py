import cv2 as cv


def Compare(img1, img2):
    img1Gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2Gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    orb = cv.ORB_create(nfeatures=1000)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    kp1, des1 = orb.detectAndCompute(img1Gray, None)
    kp2, des2 = orb.detectAndCompute(img2Gray, None)

    matches = bf.match(des1, des2)
    matches = sorted(matches,key=lambda x:x.distance)
    
    ORB_matches =cv.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=2)

    return len(matches)


cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    face_lib = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_lib.detectMultiScale(gray, 1.3, 5)

    for face in faces:
        for (x, y, w, h) in faces:
            faceFrame = frame[y:y+h, x:x+w]
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

    cv.imshow("Video", frame)

cap.release()
cv.destroyAllWindows()