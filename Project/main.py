import cv2 as cv


cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read() 
    frame = cv.flip(frame, 1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  
    face_lib = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_lib.detectMultiScale(gray, 1.3, 5)

    for face in faces:
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 4)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break
  
    cv.imshow("Video", frame)

cap.release()
cv.destroyAllWindows()
