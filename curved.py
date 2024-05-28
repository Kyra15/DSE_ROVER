# importing packages and modules
import pandas as pd
import numpy as np
import cv2
import math

def thing(image):
  def contours_2(image, og, extra_pix=0):
      # find the contours on the image
      contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      # contours = imutils.grab_contours((contours, hierarchy))
      # sort the list of contours by the contour area
      new_lst = list(contours)
      new_lst.sort(key=cv2.contourArea)
      # if there are at least 2 contours that have been detected
      if len(new_lst) > 1:
          # get the 2 largest contours
          c1 = new_lst[-1]
          c2 = new_lst[-2]
          # fit polylines to each contour
          outline1 = cv2.approxPolyDP(c1, 4, True)
          cv2.drawContours(image, [outline1], -1, (0, 255, 255), 15)
          outline2 = cv2.approxPolyDP(c2, 4, True)
          cv2.drawContours(image, [outline2], -1, (0, 255, 255), 15)
          # draw a midline by going through the polyline and averaging each x and y coordinate
          # append this averaged coordinate to a list and turn that list into a numpy array
          midline = []

          for pt1, pt2 in zip(outline1[:int(len(outline1) / 1.8)], outline2[:int(len(outline2) / 1.8)]):
              mid_x = int((pt1[0][0] + pt2[0][0]) / 2) + extra_pix
              mid_y = int((pt1[0][1] + pt2[0][1]) / 2)
              midline.append([[mid_x, mid_y]])
          midline = np.array(midline, dtype=np.int32)
          # draw a polyline from the numpy array onto the frame
          cv2.polylines(og, [midline], False, (0, 255, 0), 15)
          return midline
      cv2.imshow("mid", og)

  def colorr(image, lower, upper):
      mask = cv2.inRange(image, lower, upper)
      masked = cv2.bitwise_and(image, image, mask=mask)
      return masked

  def filtering(image):
      mask2 = colorr(image, (50, 50, 60), (250, 250, 250))
      image = image - mask2
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      gauss = cv2.GaussianBlur(gray, (15, 15), 0)
      gauss = cv2.medianBlur(gauss, 15)
      canny2 = cv2.Canny(gauss, 30, 60)
      cv2.imshow("EEEEEE", image)
      cv2.imshow("EEEEEeE", canny2)
      return canny2
    
  # un-warps an image given a set of vertices
  def unwarped(img, mask_vert, screen_vert):
      matrix2 = cv2.getPerspectiveTransform(screen_vert, mask_vert)
      result = cv2.warpPerspective(img, matrix2, (img.shape[1], img.shape[0]))
      return result
    
  # warps an image given a set a vertices
  def warping(image, mask_vert, screen_vert):
      matrixy = cv2.getPerspectiveTransform(mask_vert, screen_vert)
      result = cv2.warpPerspective(image, matrixy, (image.shape[1], image.shape[0]))
      return result
    
  height = image.shape[0]  # 1080
  global width
  width = image.shape[1]  # 1920
  p1 = [round(width * .1), round(height * 1)]
  p2 = [round(width * .22), round(height * .28)]
  p3 = [round(width * .79), round(height * .28)]
  p4 = [round(width * .90), round(height * 1)]
  # create a trapezoidal mask around the road
  mask_vertices = np.int32([p1, p2, p3, p4])
  # cv2.polylines(image, [mask_vertices], True, (0,0,0), 5)
  screen_verts = np.float32([[0, height], [0, 0], [width, 0], [width, height]])
  # warp the frame to fit this trapezoidal mask to get a bird's-eye view of the road
  warped_image = warping(image, np.float32(mask_vertices), screen_verts)
  filtered = filtering(warped_image)
  crop_l = filtered[0:height, 0:width//2]
  cv2.imshow("cropl", crop_l)
  crop_r = filtered[0:height, width//2:width]
  cv2.imshow("cropr", crop_r)
  leftc = contours_2(crop_l, warped_image)
  rightc = contours_2(crop_r, warped_image, width//2)
  middle = []
  scalell = []
  maxx = width//2
  maxy = 0
  if leftc is not None and rightc is not None:
      for x in range(len(leftc)):
          try:
              scalel =int((leftc[x][0][0] + rightc[x][0][0])/2) / int((leftc[x][0][0]))
              middle.append([int((leftc[x][0][0] + rightc[x][0][0])/2),int((leftc[x][0][1] + rightc[x][0][1])/2)])
                scalell.append(scalel)
          except:
              if scalell != []:
                  scaless = sum(scalell)/len(scalell)
                  middle.append([int((leftc[x][0][0])*scaless), int((leftc[x][0][1])*scaless)])
              else:
                  break
      for point in middle:
          if point[1] > maxy:
              maxx = point[0]
      middle = np.array(middle, dtype=np.int32)
      cv2.polylines(warped_image, [middle], False, (0, 255, 255), 15)

  unwarped = unwarped(warped_image, np.float32(mask_vertices), screen_verts)
  cv2.imshow("unwarped", unwarped)
  # add the unwarped image and the orginal image ontop of each other
  finished = cv2.addWeighted(image, 0.5, unwarped, 0.5, 0.0)
  cv2.imshow("finished", finished)
  return maxx
  
while True:
  ret, frame = vid.read()
  if frame is None:
      break
  midd = thing(frame)
  if midd < width // 2:
      print("turn left")
  elif midd == width // 2:
      print("forward")
  else:
      print("turn right")

  if cv2.waitKey(15) & 0xFF == ord('q'):
      break
vid.release()
cv2.destroyAllWindows()
