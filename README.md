# FACE MASK DETECTION USING CNN

*Created by: Bareera Mushthak*

This project aims to classify whether a person is wearing a face mask or not using Convolutional Neural Networks (CNN). It uses real-world facial images from varied angles, lighting, and occlusions. The model handles binary classification with pre processing, data augmentation, ,evaluation using confusion matrix ,classification report, and visualization to improve accuracy and interpretability.
Challenges addressed while doing this project include imbalanced data and visual diversity in faces.
---

## Dataset

The dataset was *manually downloaded* from Kaggle and contains thousands of real-world face images categorized into two classes:

- *With Mask*
- *Without Mask*

### Dataset Handling:
- Uploaded manually into Google Colab
- Extracted using Python's zipfile module
- Loaded using image_dataset_from_directory()
- Split into training and validation sets using *validation_split=0.2*
- Labels automatically inferred based on folder names
- Batch size: 32, image size: 180x180

---

## Preprocessing and Augmentation

### Normalization:
- All images were normalized using:
```python
Rescaling(1./255)

Data Augmentation:

To enhance generalization and avoid overfitting, the following augmentations were applied during training:
	•	RandomFlip("horizontal")
	•	RandomRotation(0.1)

These were implemented using a tf.keras.Sequential augmentation layer within the model.

⸻

Model Architecture

The model was built using TensorFlow Keras Sequential API, with the following layers:

Input Layer: (180, 180, 3)
↓
Data Augmentation Layer
↓
Conv2D(32, (3x3), activation='relu') + MaxPooling2D(2x2)
↓
Conv2D(64, (3x3), activation='relu') + MaxPooling2D(2x2)
↓
Conv2D(128, (3x3), activation='relu') + MaxPooling2D(2x2)
↓
Flatten
↓
Dense(128, activation='relu') + Dropout(0.5)
↓
Dense(1, activation='sigmoid') → Binary Output (Mask / No Mask)



⸻

Model Compilation and Training
	•	Optimizer: Adam
	•	Loss Function: Binary Crossentropy
	•	Metric: Accuracy
	•	Epochs: 15

The model was compiled and trained using:

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_dataset, epochs=15, validation_data=val_dataset)

Model saved as:

model.save("facemask_model.h5")



⸻

Evaluation and Output
	•	A confusion matrix was created using sklearn.metrics.confusion_matrix
	•	Plotted using seaborn to visualize correct and incorrect predictions
	•	Sample test predictions were also displayed with actual vs predicted labels

⸻

This project showcases:
	•	Manual dataset upload and organization
	•	Validation split using validation_split
	•	Data augmentation to improve generalization
	•	CNN model building from scratch with dropout
	•	Evaluation with confusion matrix and test visualizations

⸻

How to Run the Project
	1.	Open the project in Google Colab
  2.  Download the datase from kaggle
	3.	Upload the zip dataset from your local computer
	4.	Extract using Python zipfile as shown in the notebook
	5.	Run all notebook cells from top to bottom
	6.	Make sure these libraries are available:
	•	tensorflow, numpy, matplotlib, seaborn, scikit-learn

⸻

Project Files
	•	Notebook: View on GitHub
	•	Dataset Source: Face Mask Dataset (Kaggle)

⸻

Connect With Me
	•	LinkedIn: www.linkedin.com/in/bareera-mushthak
	•	GitHub: https://github.com/baree-tech
⸻

Thank you for visiting this repository!

---
