import torch
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, save_every=10, save_dir="saved_models", save_name="scTRaCT_checkpoint"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        y_true = []
        y_pred = []

        loop = tqdm(train_loader, leave=False, desc=f"Training Epoch {epoch}")
        for X_counts_batch, X_dist_batch, y_batch in loop:
            X_counts_batch, X_dist_batch, y_batch = X_counts_batch.to(device), X_dist_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_counts_batch, X_dist_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(y_true, y_pred)
        train_f1 = f1_score(y_true, y_pred, average="macro")

        # Print nicely
        print(f"\n[ Train | {epoch:03d}/{num_epochs:03d} ] loss = {avg_loss:.5f}, acc = {train_acc:.5f}, f1 = {train_f1:.5f}")

        # Save checkpoint
        if epoch % save_every == 0:
            save_path = os.path.join(save_dir, f"{save_name}_epoch{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at {save_path}")


def evaluate_model(model, test_loader, label_encoder):
    device = next(model.parameters()).device
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_counts_batch, X_dist_batch, y_batch in test_loader:
            X_counts_batch, X_dist_batch = X_counts_batch.to(device), X_dist_batch.to(device)
            outputs = model(X_counts_batch, X_dist_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu())
            all_labels.append(y_batch.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = accuracy_score(all_labels.numpy(), all_preds.numpy())
    f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average="macro")
    
    return acc, f1
