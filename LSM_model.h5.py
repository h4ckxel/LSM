import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Definir el número de clases
num_classes = 10  # Reemplaza 10 con el número real de gestos que estás reconociendo

# Definir el modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Suponiendo que tienes train_images, train_labels, test_images, y test_labels preprocesados
# Entrenar el modelo
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels)) # ERROR

# Guardar el modelo
model.save('LSM_model.h5')
