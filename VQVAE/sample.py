from VQVAE import VQVAETrainer, get_vqvae2
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
image = cv2.imread('/home/user/data/prior_train/gt/1.3.6.1.4.1.14519.5.2.1.6279.6001.979083010707182900091062408058_56.png.png', cv2.IMREAD_GRAYSCALE)
image_np = np.array(image)

# Create a new model instance
model = get_vqvae2(latent_dim=32, num_embeddings=1024)
print(model.summary())
output = model.predict([image_np])
loss, acc = model.evaluate(output, image_np, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# Load the previously saved weights

model.load_weights('checkpoints_healthy/vqvae2_healthy.ckpt')
output = model.predict(image_np)

# print("here we go")

# Re-evaluate the model
# Postprocess predictions
# You can manipulate the predictions according to your task

# Print the predictions
plt.imshow(output, cmap='gray')  # You can choose a different colormap if desired
plt.colorbar()  # Add a colorbar for reference
plt.show()