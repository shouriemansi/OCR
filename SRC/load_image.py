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
    cv2.imshow("My Digit(ORIGINAL)", image)


    # cv2.cvtColor() is used to convert an image from one color representation to another. COLOR_BGR2GRAY tells OpenCV to take the BGR color image and convert it into a grayscale image. A grayscale image has only 1 channel containing intensity(brightness) information.For digit recognition, we usually don't need the color of the digit. We mainly care about the shape and brightness of the digit, so we are converting it into grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    print("Grayscale shape:", gray.shape)


    cv2.imshow("My Digit(Grayscale)", gray)


     # Apply thresholding to the grayscale image.
     #Thresholding is used to separate the image in two regions: 1. BACKGROUND , 2. DIGIT/FOREGROUND

     # The basic logic of THRESH_BINARY is:
     # pixel < 127 → 0 (black) and pixel >= 127 → 255 (white). So instead of having many shades of gray (0–255), we convert the image into mainly two values: 0 = black  255 = white. This makes the digit easier for later processing and machine-learning steps to work with.
    _, threshold = cv2.threshold(
        gray,   #Input image(must be grayscale)
        127,    #Threshold value    
        255,    # Maximum value assigned to pixels that pass the threshold
        cv2.THRESH_BINARY   # Thresholding method )
    )


    print("Thresholding completed!")

    print("Thresholded", threshold)


    # Resize the thresholded image
    # Our thresholded image may have a different size from the images used by the MNIST dataset. A machine-learning model expects its input to have a # consistent size. Therefore, we resize our image to exactly 28 × 28 pixels so that it has the same spatial dimensions as MNIST images. 
    resized = cv2.resize(threshold, (28, 28))

    
    print("Resized shape:", resized.shape)

    cv2.imshow("Resized", resized)


    # Keep the image window open until the user presses a key
    # Without waitKey(), the program could finish immediately and the image window may close before we can see it. 0 means "wait indefinitely"
    cv2.waitKey(0)


    # Close all OpenCV windows after a key is pressed. This cleans up the windows created by cv2.imshow().
    cv2.destroyAllWindows()

    




