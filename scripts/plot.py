import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import numpy as np
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
import segmentation_models as sm
import matplotlib.pyplot as plt
from PIL import Image

# Define the model architecture
def create_model():
    base_model = sm.Unet('resnet34', encoder_weights=None)
    inp = Input(shape=(128, 32, 1))  # Input shape for spectrograms
    l1 = Conv2D(3, (1, 1))(inp)  # Add a Conv2D layer to convert 1 channel to 3 channels
    out = base_model(l1)  # Pass through the U-Net model
    model = Model(inp, out)  # Define the final model
    return model

# Create the model
model = create_model()

# Plot the model architecture and save it to a file
plot_model(
    model,
    to_file='model_architecture.png',  # Save the plot to a file
    show_shapes=True,
    show_layer_names=True,
    rankdir='TB',  # Top-to-bottom layout
    expand_nested=True,  # Expand nested models
    dpi=96  # Dots per inch for the image
)

# Display the saved image using matplotlib
img = Image.open('model_architecture.png')
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')  # Hide axes
plt.show()