import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Load the training and validation JSON files
train_json_file = "/home/hojat/Documents/Saxion/e_waste_dataset/embeddings/train_embedings.json"  
val_json_file = "/home/hojat/Documents/Saxion/e_waste_dataset/embeddings/valid_embedings.json"  

names = ['adapter', 'memory', 'mobile', 'laptop', 'monitor', 'mouse', 'keyboard', 'charger']

def load_data(json_file, class_names):
    features = []
    labels = []
    with open(json_file, 'r') as file:
        data = json.load(file)

    for i, class_name in enumerate(class_names):
        if class_name in data.keys():
            vectors = data[class_name]
            for vector in vectors:
                features.append(vector)
                labels.append(class_name)            
    return np.array(features), np.array(labels)

X_train, y_train = load_data(train_json_file, names)
X_val, y_val = load_data(val_json_file, names)

# Encode class labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)

# Step 2: Train classifiers

# # Random Forest Classifier
# rf_classifier = RandomForestClassifier(random_state=42)
# rf_classifier.fit(X_train, y_train_encoded)
# rf_predictions = rf_classifier.predict(X_val)

# # SVM Classifier
# svm_classifier = SVC(probability=True, random_state=42)
# svm_classifier.fit(X_train, y_train_encoded)
# svm_predictions = svm_classifier.predict(X_val)

# # Logistic Regression
# logreg_classifier = LogisticRegression(max_iter=1000, random_state=42)
# logreg_classifier.fit(X_train, y_train_encoded)
# logreg_predictions = logreg_classifier.predict(X_val)

# k-NN Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
t0 = time.time()
knn_classifier.fit(X_train, y_train_encoded)
print(f"KNN training time: {time.time()-t0:.3f}")
t0 = time.time()
knn_predictions = knn_classifier.predict(X_val)
print(f"KNN inference time: {(time.time()-t0):.3f}")

# Neural Network (PyTorch)
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Prepare PyTorch dataset
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

input_size = X_train.shape[1]
num_classes = len(np.unique(y_train_encoded))
nn_model = NeuralNet(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

t0 = time.time()
# Train Neural Network
for epoch in range(20):  # 20 epochs
    for batch_features, batch_labels in train_loader:
        outputs = nn_model(batch_features)
        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
print(f"Neural network training time: {time.time()-t0:.3f}")

# Evaluate Neural Network
with torch.no_grad():
    t0 = time.time()
    nn_outputs = nn_model(X_val_tensor)
    print(f"Neural network inference time: {(time.time()-t0):.3f}")
    _, nn_predictions = torch.max(nn_outputs, 1)
    nn_predictions = nn_predictions.numpy()

# Step 3: Evaluate all models and plot confusion matrices
def evaluate_and_plot_cm(name, y_true, y_pred, labels):
    # Get the unique labels present in the validation set
    unique_labels = np.unique(y_true)
    # Map these to their corresponding class names from the full list
    unique_class_names = [labels[i] for i in unique_labels]
    
    print(f"\n{name} Classifier:")
    print(classification_report(y_true, y_pred, digits=3, labels=unique_labels, target_names=unique_class_names, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred, normalize='true')  # Normalized to show percentages
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", xticklabels=unique_class_names, yticklabels=unique_class_names)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

class_names = label_encoder.classes_
# evaluate_and_plot_cm("Random Forest", y_val_encoded, rf_predictions, class_names)
# evaluate_and_plot_cm("SVM", y_val_encoded, svm_predictions, class_names)
# evaluate_and_plot_cm("Logistic Regression", y_val_encoded, logreg_predictions, class_names)
evaluate_and_plot_cm("k-NN", y_val_encoded, knn_predictions, class_names)
evaluate_and_plot_cm("Neural Network", y_val_encoded, nn_predictions, class_names)
plt.show()

# Step 4: Save the models and encoders
import joblib
joblib.dump(label_encoder, 'label_encoder.pkl')
# joblib.dump(rf_classifier, 'random_forest_model.pkl')
# joblib.dump(svm_classifier, 'svm_model.pkl')
# joblib.dump(logreg_classifier, 'logistic_regression_model.pkl')
joblib.dump(knn_classifier, 'knn_model.pkl')
torch.save(nn_model.state_dict(), 'neural_network_model.pth')
