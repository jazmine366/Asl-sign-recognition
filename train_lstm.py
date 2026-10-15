from pathlib import Path 
import numpy as np 
import torch #tensors & model training
import torch.nn as nn #provide LSTM and other NN layers
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import confusion_matrix, classification_report

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

#1. Load recordings and prepare as training data

data_dir = Path("v1_data") #path -> v1_data folder
labels = ["A", "J", "NAME"]

X= []
y= []

for label_index, label in enumerate(labels):
    folder = data_dir / label

    for file in sorted(folder.glob("*.npy")):
        sequence = np.load(file)

        X.append(sequence)
        y.append(label_index)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print("X shape:", X.shape)
print("y shape:", y.shape)

#2. Split dataset 70/15/15 
#2a. Train + Val: Test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size = 0.15,
    random_state=42,
    stratify=y
)

#2b. Train : Val
X_train, X_val, y_train, y_val  = train_test_split(
    X_train_val, y_train_val, 
    test_size = 0.15/0.85,
    random_state=42,
    stratify=y_train_val
)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)
#Training: Model learning pattern + update weights
#validation: Choose hyperparameters + monitor overfitting 
#Testing: Final evaluation 
#train_test_split: leftOver -> group 1, test_size -> group 2

#3. Covert NumPy arr -> PyTorch tensors
#NumPy arr: loading + manipulate data
#PyTorch tensors: similar but automatic differential in model training 
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype = torch.long)
y_test = torch.tensor(y_test, dtype = torch.long)

#Define LSTM Model 
class ASLLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        #LSTM Layer
        self.lstm = nn.LSTM(
            input_size = 132,#input features per frame
            hidden_size=64,  #input -> represent features
            batch_first=True #(batch, frames, features)
        )
        #Fully Connected lay 64 learned features -> 3 class scores
        self.fc = nn.Linear(64,3) 
    
    #How data flow through model 
    def forward(self,x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_output = lstm_out[:,-1,:] #(recording, frame, hidden features)
        scores = self.fc(last_output)
        return scores 
    
model = ASLLSTM()
criterion = nn.CrossEntropyLoss() #loss function 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#epoch = 1 complete pass in training dataset
for epoch in range(100):
    model.train()           #mode -> training mode
    optimizer.zero_grad()   #clear prev gradients
    scores = model(X_train)  #predict w current weights
    loss = criterion(scores, y_train) #wrongness
    loss.backward()         #calc gradients
    optimizer.step()        #update weights 

    model.eval()

    with torch.no_grad():
        val_scores = model(X_val)
        val_preds = val_scores.argmax(dim=1)
        val_acc = (val_preds == y_val).float().mean()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1}: "
            f"Loss = {loss.item():.4f}, "
            f"Val Accuracy = {val_acc.item():.2%}"
        )

model.eval() #model -> evaluate mode
with torch.no_grad(): #disable gradient tracking
    #training accuracy
    train_scores = model(X_train) #3 class logits 
    train_preds = train_scores.argmax(dim=1) #highest scoring class
    train_acc = (train_preds == y_train).float().mean() #Boolean -> float -> average

    #validation accuracy
    val_scores = model(X_val)
    val_preds = val_scores.argmax(dim=1)
    val_acc = (val_preds == y_val).float().mean()

print(f"Training Accuracy: {train_acc.item():.2%}")
print(f"Validation Accuracy: {val_acc.item():.2%}")

#Save to a file the model's learned parameters
torch.save(model.state_dict(), "asl_lstm_v1.pth") 
print("Model saved!")

#7. Test Evaluation 
model.eval()

with torch.no_grad():
    test_scores = model(X_test)
    test_preds = test_scores.argmax(dim=1)
    test_acc = (test_preds == y_test).float().mean()

print(f"Test Accuracy: {test_acc.item():.2%}")

# Confusion Matrix
cm = confusion_matrix(
    y_test.numpy(),
    test_preds.numpy(),
    labels=[0, 1, 2]
)

print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(
    classification_report(
        y_test.numpy(),
        test_preds.numpy(),
        labels=[0, 1, 2],
        target_names=labels,
        zero_division=0
    )
)