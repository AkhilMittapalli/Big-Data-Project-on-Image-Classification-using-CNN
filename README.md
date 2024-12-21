# Image Classification of Places 365 - README

## **Project Overview**
The project focuses on developing an end-to-end image classification pipeline using the Places 365 dataset. The pipeline leverages Azure's ecosystem, including Azure Blob Storage for data storage, Azure Databricks for distributed training and preprocessing, and Azure MLFlow for model deployment and tracking. The model classifies high-resolution images into categories representing different scenes.

---

## **Steps to Recreate the Project**

### **Step 1: Set Up Azure Environment**
#### 1. Create a Resource Group
   - Navigate to the Azure portal.
   - Go to **Resource Groups** and click **+ Create**.
   - Enter a name for the resource group (e.g., `ImageClassificationRG`) and select a region.
   - Click **Review + Create** and then **Create**.

#### 2. Create Azure Storage Account
   - In the Azure portal, go to **Storage Accounts** and click **+ Create**.
   - Select the previously created resource group.
   - Provide a unique name for the storage account (e.g., `imageclassstorage`).
   - Choose `Standard` for performance and `Hot` for access tier.
   - Click **Review + Create** and then **Create**.

#### 3. Upload Dataset to Azure Blob Storage
   - Go to the created storage account.
   - Select **Containers** and create a new container (e.g., `places365`).
   - Upload the dataset files organized by categories into the container.

---

### **Step 2: Set Up Azure Databricks Workspace**
#### 1. Create Azure Databricks Workspace
   - Go to **Azure Databricks** in the Azure portal and click **+ Create**.
   - Select the resource group created earlier.
   - Provide a workspace name (e.g., `ImageClassificationDBW`).
   - Choose the appropriate pricing tier (e.g., `Standard`).
   - Click **Review + Create** and then **Create**.

#### 2. Set Up Databricks Cluster
   - Open the Databricks workspace.
   - Go to **Clusters** and click **+ Create Cluster**.
   - Provide a name for the cluster (e.g., `ImageClassificationCluster`).
   - Choose a runtime version with ML capabilities (e.g., `Databricks Runtime 12.0 ML`).
   - Select the appropriate worker type (e.g., `Standard_DS3_v2`) and the number of workers.
   - Click **Create Cluster**.

#### 3. Mount Azure Blob Storage in Databricks
   ```python
   dbutils.fs.mount(
       source="wasbs://<container-name>@<storage-account-name>.blob.core.windows.net/",
       mount_point="/mnt/<mount-name>",
       extra_configs={"fs.azure.account.key.<storage-account-name>.blob.core.windows.net": "<storage-account-key>"}
   )
   ```
   Replace `<container-name>`, `<storage-account-name>`, `<mount-name>`, and `<storage-account-key>` with your details.

---

### **Step 3: Data Preprocessing**
#### Preprocessing Pipeline in Databricks
1. Resize images to 128x128 pixels.
2. Normalize pixel values to `[0, 1]`.
3. Store preprocessed data in-memory or save intermediate files for training.

Example Code:
```python
from PIL import Image
import numpy as np
import os

def preprocess_image(image_path, target_size=(128, 128)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    return image_array
```

---

### **Step 4: Model Training**
#### TensorFlow Implementation
- Use Convolutional Neural Networks (CNNs) for classification.
- Apply dropout layers to prevent overfitting.

Example Training Code:
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 categories
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=20)
```

---

### **Step 5: Model Deployment**
#### Azure MLFlow Integration
1. **Log Model to MLFlow**:
   ```python
   import mlflow
   import mlflow.keras

   mlflow.keras.log_model(model, "places365_model")
   ```

2. **Deploy Model as Real-Time Endpoint**:
   - In Azure ML Studio, go to **Endpoints** > **+ New Endpoint**.
   - Configure the deployment with the logged model.
   - Deploy the endpoint for real-time predictions.

3. **Test Endpoint**:
   ```python
   import requests
   import json

   url = "https://<your-endpoint>.azurewebsites.net/score"
   headers = {"Authorization": "Bearer <your-key>", "Content-Type": "application/json"}

   data = {"inputs": [[...]]}  # Replace with preprocessed image data
   response = requests.post(url, headers=headers, json=data)
   print(response.json())
   ```

---

### **Step 6: Tracking Parameters with Azure MLFlow**
1. **Enable Experiment Tracking**:
   ```python
   mlflow.set_experiment("ImageClassificationExperiment")
   ```
2. **Log Metrics and Parameters**:
   ```python
   with mlflow.start_run():
       mlflow.log_param("epochs", 20)
       mlflow.log_param("batch_size", 32)
       mlflow.log_metric("accuracy", 0.78)
       mlflow.keras.log_model(model, "model")
   ```

---

### **Step 7: Results and Visualizations**
- Achieved ~64% accuracy on the test dataset.
- Visualized accuracy vs. epochs and confusion matrix for predictions.

---

## **Future Enhancements**
1. Scale training to the full Places 365 dataset.
2. Implement batch predictions for large-scale inference.
3. Optimize CNN architecture for higher accuracy.

---

## **Contact**
For questions or issues, contact:
- **Email**: mittapalliakhil12@gmail.com
- **GitHub**: https://github.com/AkhilMittapalli/Big-Data-Project-on-Image-Classification-using-CNN

