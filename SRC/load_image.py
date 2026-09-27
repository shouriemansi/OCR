# OpenCV (cv2)= computer vision library that is used here to read, display, and work with images.
import cv2

# Load the image from DATA/raw folder
# cv2.imread() reads the image file and converts it into a NumPy array containing the image's pixel information.
image = cv2.imread("../DATA/raw/digit2.jpeg")

# If OpenCV cannot find/read the image, imread() returns None. We check this before trying to use the image.
if image is None:
    print("Image could not be loaded!")
else:
    print("Image loaded successfully!")
    print("Image shape:", image.shape)
    #Shape tells us the dimensions of the image, for a colored image, the shape is: height × width × channels(bgr)

    # Display the image in a separate window
    cv2.imshow("My Digit", image)

    # Keep the image window open until the user presses a key
    # Without waitKey(), the program could finish immediately and the image window may close before we can see it. 0 means "wait indefinitely"
    cv2.waitKey(0)

    # Close all OpenCV windows after a key is pressed. This cleans up the windows created by cv2.imshow().
    cv2.destroyAllWindows()