import pathlib
import cv2 # OpenCV

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / 'data' / 'haarcascade_frontalface_default.xml' # Path to the cascade file

print(cascade_path) 

clf = cv2.CascadeClassifier(str(cascade_path)) # Load the cascade

camera = cv2.VideoCapture(0) # Open the camera

while True:
    ret, frame = camera.read() # Capture frame-by-frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    faces = clf.detectMultiScale(gray, 1.1, 4) # Detect the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Draw the rectangle around each face
    cv2.imshow('Video', frame) # Display
    if cv2.waitKey(1) & 0xFF == ord('q'): # Break if 'q' is pressed
        break
    
    
camera.release() # When everything is done, release the capture
cv2.destroyAllWindows() # And destroy the windows


