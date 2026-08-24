import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "preprocessed_fatigue_data.npz")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "best_fatigue_model.pth")


# 1. Define the Neural Network Architecture
class FatigueMLP(nn.Module):
    def __init__(self, input_dim=180, num_classes=4):
        super(FatigueMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # Prevents overfitting
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


def main():
    if not os.path.exists(DATA_PATH):
        print("❌ Error: Preprocessed data not found. Run preprocessing first.")
        return

    # 2. Load Preprocessed Data
    data = np.load(DATA_PATH)
    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    X_test = torch.tensor(data["X_test"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.long)
    y_test = torch.tensor(data["y_test"], dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 3. Instantiate Model, Loss, Optimizer
    model = FatigueMLP(input_dim=180, num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Training Loop
    epochs = 50
    print("🚀 Starting PyTorch Training (50 Epochs)...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        train_acc = (correct / total) * 100
        epoch_loss = total_loss / total

        if epoch % 10 == 0 or epoch == 1:
            print(f"   Epoch [{epoch:2d}/{epochs}] - Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}%")

    # 5. Evaluate on Unseen Test Set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, test_preds = torch.max(test_outputs, 1)

    test_acc = (test_preds == y_test).sum().item() / len(y_test) * 100

    print("\n==============================")
    print("📊 FINAL MODEL EVALUATION")
    print("==============================")
    print(f"Test Set Accuracy: {test_acc:.2f}%\n")

    class_names = ["alert", "distracted", "drowsy", "yawning"]
    print("Classification Report:")
    print(classification_report(y_test.numpy(), test_preds.numpy(), target_names=class_names))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test.numpy(), test_preds.numpy()))

    # 6. Save Model Weights
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n💾 Model successfully saved to:\n   {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()