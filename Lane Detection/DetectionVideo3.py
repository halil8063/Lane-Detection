import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

kamera = cv2.VideoCapture('video3.mp4')
ysize = int(kamera.get(3))
xsize = int(kamera.get(4))

low_threshold = 300
high_threshold = 500
kernel_size = 1

while True:

    ret1,kare = kamera.read()
    if not ret1:
        break

    gray = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    mask = np.zeros_like(edges)
    ignore_mask_color = 255

    imshape = kare.shape
    vertices = np.array([[(50, 1500), (650,550), (660, 550), (4500, 1500)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 1  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 2  # minimum number of pixels making up a line
    max_line_gap = 10  # maximum gap in pixels between connectable line segments
    line_image = np.copy(kare) * 1

    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    # Iterate over the output "lines" and draw lines on a blank image
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 15)

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges))

    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)

    cv2.imshow("Duzgun",lines_edges)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


kamera.release()
cv2.destroyAllWindows()