import tensorflow as tf
import matplotlib.pyplot as plt

# Set random seed for reproducibility
tf.keras.utils.set_random_seed(12)

# Define constants
img_height, img_width = 224, 224
batch_size = 32

# Define directory paths
train_dir = r"C:\Users\NEHA\Downloads\pcos\data\train"
test_dir = r"C:\Users\NEHA\Downloads\pcos\data\test"

# Load training and validation datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="binary",
    shuffle=True,
    seed=12,
    validation_split=0.15, 
    subset="training",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="binary",
    shuffle=True,
    seed=12,
    validation_split=0.15,
    subset="validation",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Load test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Visualize a batch of images
class_names = train_ds.class_names
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        label_index = int(labels[i].numpy())
        plt.title(class_names[label_index])
        plt.axis("off")
plt.show()

# Normalize pixel values to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Apply normalization to datasets
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Build the model using a pre-trained base model (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)


# Evaluate the model safely — avoid error if test_ds is empty
if tf.data.experimental.cardinality(test_ds).numpy() == 0:
    print("Test dataset is empty. Please add test images!")
else:
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc * 100:.2f}%")
model.save("pcos_ultrasound_model.h5")
print("Model saved successfully!")
