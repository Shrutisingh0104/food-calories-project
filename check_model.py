import tensorflow as tf

model = tf.keras.models.load_model("model.h5")

print("\nMODEL INPUT SHAPE:")
print(model.input_shape)

print("\nMODEL OUTPUT SHAPE:")
print(model.output_shape)

print("\nMODEL SUMMARY:")
model.summary()