import cv2


cap = cv2.VideoCapture(0)

while True:
  ret, image = cap.read()
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  face_lib = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
  
  faces = face_lib.detectMultiScale(gray, 1.3, 5)

  for face in faces:
    for (x, y, w, h) in faces:
      cv2.rectangle(image, (x,y), (x+w,y+h), (255, 0, 0), 4)

  if cv2.waitKey(10) & 0xFF == ord('q'):
    break
  
  cv2.imshow("Video", image)

cap.release()
cv2.destroyAllWindows()
