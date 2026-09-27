import cv2

# Load the image
image = cv2.imread("../DATA/raw/digit2.jpeg")

if image is None:
    print("Image could not be loaded!")
else:
    print("Image loaded successfully!")
    print("Image shape:", image.shape)

    # Display the image
    cv2.imshow("My Digit", image)

    # Wait until we press a key
    cv2.waitKey(0)

    # Close the image window
    cv2.destroyAllWindows()