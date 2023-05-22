import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# Load pre-trained models
content_image_path = "C:\\Users\\RAJATH H S\\Desktop\\images-1.jpg"
style_image_path = "C:\\Users\\RAJATH H S\\\Desktop\\background.jpg"

content_image = plt.imread(content_image_path)
style_image = plt.imread(style_image_path)

# Resize images to match the expected shape
content_image = tf.image.resize(content_image, (256, 256))
style_image = tf.image.resize(style_image, (256, 256))

# Convert images to float32 and normalize
content_image = tf.cast(content_image, tf.float32) / 255.0
style_image = tf.cast(style_image, tf.float32) / 255.0

# Add batch dimension
content_image = content_image[tf.newaxis, ...]
style_image = style_image[tf.newaxis, ...]

# Load the pre-trained models from TensorFlow Hub
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Perform style transfer
outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
stylized_image = outputs[0]

# Display the stylized image
plt.imshow(stylized_image.numpy())
plt.axis('off')
plt.show()
