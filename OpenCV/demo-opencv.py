import subprocess
import cv2  
import random
 
# build our cv2 Cascade Classifiers
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
noseCascade = cv2.CascadeClassifier("haarcascade_nose.xml")
 
#-----------------------------------------------------------------------------
#       Take a photo
#-----------------------------------------------------------------------------
def take_a_picture():
  cap = cv2.VideoCapture(0)
  ret, frame = cap.read()
  cv2.imwrite("original.jpg", frame)
  #subprocess.call("fbi -T 2 original.jpg", shell = True)
  cap.release()

#-----------------------------------------------------------------------------
#       Face Detection and Add mustache
#-----------------------------------------------------------------------------
def face_detection():
  frame = cv2.imread("original.jpg", -1)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  height, width, channels = frame.shape
  print height, width, channels

  faces = faceCascade.detectMultiScale(
          gray,
          scaleFactor=1.1,
          minNeighbors=5,
          minSize=(30, 30),
          flags=cv2.cv.CV_HAAR_SCALE_IMAGE
       )
  for (x, y, w, h) in faces:
    face = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]

    nose = noseCascade.detectMultiScale(roi_gray)
    for (nx,ny,nw,nh) in nose:
      # Un-comment the next line for debug (draw box around the nose)
      nose = cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,255,0),2)

      #-----------------------------------------------------------------
      # Load and configure mustache
      #-----------------------------------------------------------------
      num = random.randint(0, 5)
      filename = 'mustache/mo' + str(num) + '.png'
      print filename

      # Load overlay image mustache
      imgMustache = cv2.imread(filename,-1)
      # Create the mask for the mustache
      #orig_mask = cv2.threshold(imgMustache, 10, 255, cv2.THRESH_BINARY)
      orig_mask = imgMustache[:,:,3]
      # Create the inverted mask for the mustache
      orig_mask_inv = cv2.bitwise_not(orig_mask)

      # Convert mustache image to BGR
      # and save the original image size (used later when re-sizing the image)
      imgMustache = imgMustache[:,:,0:3]
      origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]

      # The mustache should be three times the width of the nose
      mustacheWidth =  2 * nw
      mustacheHeight = mustacheWidth * origMustacheHeight / origMustacheWidth

      # Center the mustache at the bottom of the nose
      x1 = nx - (mustacheWidth/4)
      x2 = nx + nw + (mustacheWidth/4)
      y1 = ny + nh - (mustacheHeight/2)
      y2 = ny + nh + (mustacheHeight/2)
      print x1,x2,y1,y2

      # Check for clipping
      if x1 <  0:
        x1 = 0
      if y1 < 0:
        y1 = 0
      if x2 > w:
        x2 = w
      if y2 > h:
        y2 = h

      # Re-calculate the width and height of the mustache image
      mustacheWidth = x2 - x1
      mustacheHeight = y2 - y1
   
      # Re-size the original image and the masks to the mustache sizes
      # calcualted above
      mustache = cv2.resize(imgMustache, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
      mask = cv2.resize(orig_mask, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
      mask_inv = cv2.resize(orig_mask_inv, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)

      # take ROI for mustache from background equal to size of mustache image
      roi = roi_color[y1:y2, x1:x2]
      # roi_bg contains the original image only where the mustache is not
      # in the region that is the size of the mustache.
      roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
      # roi_fg contains the image of the mustache only where the mustache is
      roi_fg = cv2.bitwise_and(mustache,mustache,mask = mask)

      # join the roi_bg and roi_fg
      dst = cv2.add(roi_bg,roi_fg)
      roi_color[y1:y2, x1:x2] = dst
      break

  cv2.imwrite("mockup.jpg", frame)    
#  subprocess.call("fbi -T 2 mockup.jpg", shell=True)

#-----------------------------------------------------------------------------
#       Main program loop
#-----------------------------------------------------------------------------
#take_a_picture()
face_detection()
