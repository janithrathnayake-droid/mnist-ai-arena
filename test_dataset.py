import numpy as np
import matplotlib.pyplot as plt
import random

# Load dataset
data = np.load("data/mnist_test.npz")

X_test = data["X_test"]
y_test = data["y_test"]

# Select a random image
index = random.randint(0, len(X_test) - 1)

image = X_test[index].reshape(28, 28)
label = y_test[index]

# Display image
plt.imshow(image, cmap="gray")
plt.title(f"Actual Digit: {label}")
plt.axis("off")
plt.show()