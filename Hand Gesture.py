import numpy as np
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

train_dir = r"C:\Users\kusum\OneDrive\Desktop\Prodigy InfoTech Internship\PRODIGY_ML_04-main\PRODIGY_ML_04-main\Train"
test_dir = r"C:\Users\kusum\OneDrive\Desktop\Prodigy InfoTech Internship\PRODIGY_ML_04-main\PRODIGY_ML_04-main\Test"   

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 
           'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Space', 'Nothing']

img_size = (224, 224)

train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2 
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=8,
    class_mode='sparse',
    subset='training' 
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=8,
    class_mode='sparse',
    subset='validation' 
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(classes), activation='softmax')  
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, epochs=10, validation_data=val_generator)

val_loss, val_accuracy = model.evaluate(val_generator)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

test_images = os.listdir(test_dir)

def preprocess_test_image(img_path):
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  
    return img_array

predictions = []

for img_name in test_images:
    img_path = os.path.join(test_dir, img_name)
    img_array = preprocess_test_image(img_path)
    
    pred = model.predict(img_array)
    predicted_class_index = np.argmax(pred)
    predicted_class_label = classes[predicted_class_index] 
    predictions.append([img_name, predicted_class_label])

submission_df = pd.DataFrame(predictions, columns=['filename', 'label'])
submission_df.to_csv('submission.csv', index=False)

print("Predictions saved to submission.csv")