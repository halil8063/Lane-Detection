import matplotlib.pyplot as plt
import numpy as np
import cv2

# Define color selection criteria
# MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION
red_threshold = 140
green_threshold = 230
blue_threshold = 140

rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Define the vertices of a triangular mask.
# Keep in mind the origin (x=0, y=0) is in the upper left
# MODIFY THESE VALUES TO ISOLATE THE REGION
# WHERE THE LANE LINES ARE IN THE IMAGE
left_bottom = [50, 700]
right_bottom = [1230, 700]
apex = [640, 500]

# Perform a linear fit (y=Ax+B) to each of the three sides of the triangle
# np.polyfit returns the coefficients [A, B] of the fit
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (720, 720), 1)

# Display the image and show region and color selections
x = [left_bottom[0] + 10, right_bottom[0], apex[0], left_bottom[0] + 10, 10, 1000, 1000]
y = [left_bottom[1] + 10, right_bottom[1], apex[1], left_bottom[1] + 10, 720, 720, 425]
# plt.plot(x, y, 'b--', lw=4)

kamera = cv2.VideoCapture('video2.mp4')
ysize = int(kamera.get(3))
xsize = int(kamera.get(4))

XX, YY = np.meshgrid(np.arange(0, ysize), np.arange(0, xsize))
region_thresholds = (YY > (XX * fit_left[0] + fit_left[1])) & \
                    (YY > (XX * fit_right[0] + fit_right[1])) & \
                    (YY < (XX * fit_bottom[0] + fit_bottom[1]))
ret1, kare = kamera.read()

cv2.imwrite("deneme1.jpg", kare)

while True:
    ret1, kare = kamera.read()
    # Grab the x and y size and make a copy of the image

    color_select = np.copy(kare)
    line_image = np.copy(kare)

    color_thresholds = (kare[:, :, 0] < rgb_threshold[0]) | \
                       (kare[:, :, 1] < rgb_threshold[1]) | \
                       (kare[:, :, 2] < rgb_threshold[2])

    color_select[color_thresholds | ~region_thresholds] = [0, 0, 0]
    line_image[~color_thresholds & region_thresholds] = [255, 0, 0]

    gray = cv2.cvtColor(line_image, cv2.COLOR_RGB2GRAY)
    kernel_size = 3  # Must be an odd number (3, 5, 7...)
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Define our parameters for Canny and run it
    low_threshold = 10
    high_threshold = 100
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    cv2.imshow("Duzgun", line_image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()
