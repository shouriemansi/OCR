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

    




### OCR Image Preprocessing with OpenCV
#This part of the OCR project uses OpenCV (cv2) to preprocess a handwritten digit image before it is given to a machine-learning model.
#The preprocessing pipeline consists of the following steps:

#1. Load the image: The digit image is loaded from the `DATA/raw` folder using cv2.imread().

#2. Check whether the image was loaded successfully: The program checks whether OpenCV returned a valid image. If the image cannot be found or read, an error message is displayed.

#3. Display the original image: The original digit image is displayed using cv2.imshow().

#4. Convert the image to grayscale: The BGR image is converted into a grayscale image using cv2.cvtColor(). This removes unnecessary color information while preserving the brightness and shape information needed for digit recognition.

#5. Apply thresholding: Binary thresholding is applied to separate the digit from the background. Pixel values below the threshold are converted to 0, while values above the threshold are converted to 255. This produces a simplified black-and-white image.

#6. Resize the image: The thresholded image is resized to 28 × 28 pixels using cv2.resize(). This matches the spatial dimensions expected by the MNIST dataset, which will later be used for training the digit-recognition model.

#7. Display the processed image: The resized image is displayed so that the preprocessing result can be visually inspected.

#8. Keep the windows open and close them properly: cv2.waitKey(0) keeps the OpenCV windows open until a key is pressed, while cv2.destroyAllWindows() closes the windows afterward.

#Processing Pipeline : Original Image → Grayscale → Threshold → Resize to 28×28 → Ready for ML Model

#This preprocessing converts a real-world handwritten digit image into a standardized format that can later be passed to a machine-learning model for digit classification.
#Currently testing the preprocessing pipeline using a single sample image.
# Later, this pipeline will be made reusable for any input digit.