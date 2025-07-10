import numpy as np
import tensorflow as tf 
from modelstruct import build_model
from tensorflow.keras.preprocessing import image_dataset_from_directory

dataset = "assets/archive/TB_Chest_Radiography_Database"
Img_Size = (96, 96) 

# Loading Training Dataset
train_dataset = image_dataset_from_directory(
    dataset,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=Img_Size, 
    batch_size=32
)

# Loading Validation Dataset
val_dataset = image_dataset_from_directory(
    dataset,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=Img_Size, 
    batch_size=32
)

# Building Model
modelstruct = build_model()

# Training Model
modelstruct.fit(train_dataset, epochs=10, validation_data=val_dataset)

model_json = modelstruct.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Saving model weights
modelstruct.save_weights("model_weights.h5")

