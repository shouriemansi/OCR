#MNIST gives us thousands of handwritten digit images. Each image is stored as 784 numbers (pixels), and each image has a correct answer/label. MNIST is one of the most commonly used datasets for learning handwritten digit recognition.


#scikit-learn (sklearn) is a popular Python machine-learning library. fetch_openml() allows us to download/fetch datasets that are available on OpenML. We use it here to obtain the MNIST handwritten digit dataset. 
from sklearn.datasets import fetch_openml


# matplotlib is used to display images and create graphs.
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression

print("Loading MNIST...")

#Fetch the MNIST dataset from OpenML.
mnist = fetch_openml(
    "mnist_784",     #This is the name of the MNIST dataset on OpenML.

    version=1,       #This specifies the version of the dataset we want to use.

    as_frame=False   #This tells scikit-learn NOT to return the data as a pandas DataFrame. Instead, we get NumPy-style arrays, which are convenient for machine-learning operations.
)

# Because many traditional ML algorithms expect each example to be represented as a single row of features.Flattening is basically converting the image from a visual format into a format that a traditional ML model can directly use as input. Since: 28 × 28 = 784. one MNIST image can be represented as: 28 × 28 image → 784 pixel values
print("MNIST loaded!")


# Images: the actual image pixel data
X = mnist.data

# Correct answers: the correct digit/answer for each image
y = mnist.target


# Look at the first image. X contains all the images from the MNIST dataset.
print("First image:")
print(X[0])

print()

# Look at its answer, y contains the correct labels/answers for all MNIST images.
print("Correct answer:")
print(y[0])


# Convert the 784 numbers back into a 28 × 28 image, reshape() DOES NOT change the pixel values. It only changes how the values are arranged.
image = X[0].reshape(28, 28)
print("Image shape:", image.shape)


# Look at the first 5 rows and first 5 columns
print("First 5 × 5 pixels:")
print(image[:5, :5])


# Get the pixel value at row 0, column 0
print("One pixel value:", image[0, 0])


# Show the image using Matplotlib plt.imshow() takes the 28 × 28 array of pixel values and displays it as an image. cmap="gray" tells Matplotlib to display the image using shades of gray instead of colors. 
plt.imshow(image, cmap="gray")
plt.title("MNIST Image")


# Display the image on the screen. plt.show()
plt.show()

# imshow() = what to show, takes your data and prepares it as an image.
# show() = actually show it, actually displays the prepared plot/image.


# How many images do we have?
print("Number of images:", X.shape[0])


# How many pixels does each image have?
print("Pixels per image:", X.shape[1])


# Look at the first 5 correct answers
print("First 5 labels:", y[:5])


# This code loads the MNIST handwritten digit dataset from OpenML using scikit-learn. MNIST contains 70,000 handwritten digit images, where each image is 28 × 28 pixels. Each image contains 784 pixel values because 28 × 28 = 784.
# When the dataset is loaded, these 28 × 28 images are stored as flattened arrays containing 784 numbers.
# X stores the pixel values (input/features), while y stores the correct digit labels (answers) for those images.
# X[0] gives the first image's 784 pixel values, and y[0] gives the correct digit for that image.
# The first image is reshaped from 784 values back into a  28 × 28 grid using reshape(28, 28). This does not change the pixel values; it only changes their arrangement so the image can be displayed.
# The code then examines some pixel values and displays the image using Matplotlib. imshow() prepares the pixel array  to be displayed as an image, while show() actually displays the figure.
# Overall:
# MNIST DATASET
#      ↓
#  X = pixel values
#  y = correct labels
#      ↓
#  X[0] = first image (784 pixels)
#      ↓
#  reshape(28, 28)
#      ↓
#  28 × 28 image
#      ↓
#  display using Matplotlib

# The main idea is that a handwritten image is converted into numerical pixel values, and these values will later be given to a Machine Learning model so that it can learn to recognize which digit the image represents.





# Logistic Regression is our first Machine Learning algorithm for recognizing handwritten digits. We use Logistic Regression to learn the relationship between: X → pixel values of the handwritten images and y → correct digit labels (0 to 9). At this point, the model has NOT learned anything yet. We are only creating the model.


# max_iter specifies the maximum number of times the algorithm is allowed to go through its optimization process while trying to find the best parameters for the model. We use 100 iterations here.

model = LogisticRegression(
    max_iter=100
)

print("Model created!")


# TRAIN THE MODEL : Training means allowing the Machine Learning model to learn patterns from our examples.

# fit() tells the model to learn from our examples.
# The model looks at X and y and learns patterns that can later be used to recognize digits.
# After fit() finishes, the model has learned from the # training data and can be used to predict digits.

print("Training started...")

model.fit(X, y)

print("Training completed!")

# These lines of code say: Create a Logistic Regression learner. Then give that learner the MNIST images (X) and their correct answers (y). Let it learn mathematical patterns from those examples. Once training is finished, we have a trained model that can be used to predict digits from new images.



# MAKE OUR FIRST PREDICTION

# X[0] contains the 784 pixel values of the first image.
first_image = X[0]

# The model expects: number of images × number of features
#The trained Logistic Regression model expects the input data to have this structure: number of images × number of features. For MNIST:number of features = 784. Currently, first_image contains: 784 values. Its shape is: (784,) But the model expects a 2D structure: (number of images, number of features) Since we are giving the model only ONE image: 1 image × 784 features Therefore, we reshape: (784,) → (1, 784) The -1 tells NumPy to automatically calculate the remaining dimension.  Since there are 784 values and we specify 1 row:784 ÷ 1 = 784 So the final shape becomes: # # (1, 784)
first_image = first_image.reshape(1, -1)

# Ask the trained model to predict the digit.
prediction = model.predict(first_image)

# Display the prediction.
print("Predicted digit:", prediction[0])

# Display the actual/correct answer.
print("Actual digit:", y[0])