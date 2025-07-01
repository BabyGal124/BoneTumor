# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the Dataset
df = pd.read_csv('bone_tumor_dataset.csv')  # If you rename it
  # Change this to your dataset path
print("Dataset loaded successfully!")
print("First 5 rows of the dataset:\n", df.head())

# Step 3: Data Preprocessing
# Check for missing values
print("Checking for missing values:\n", df.isnull().sum())

# Handle missing values (if any)
df = df.fillna(df.mean())  # Replace missing values with column mean

# Encode categorical values (if needed)
df['diagnosis'] = df['diagnosis'].map({'benign': 0, 'malignant': 1})

# Step 4: Feature Scaling
X = df.drop('diagnosis', axis=1)  # Features (all columns except 'diagnosis')
y = df['diagnosis']  # Target variable

# Standardize the feature columns
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 5: Split the Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Evaluate the Model
y_pred = model.predict(X_test)

# Print accuracy and classification report
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Optional: Plot confusion matrix for more insights
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()  # This will ensure the graph shows in VS Code
