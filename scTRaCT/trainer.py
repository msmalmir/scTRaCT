import torch
import os

def train_model(
    model, train_loader, val_loader, 
    criterion, optimizer, num_epochs=50, 
    save_dir="./", save_every=10, save_name="model_checkpoint"
):
    device = next(model.parameters()).device
    best_val_loss = float("inf")

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_counts_batch, X_dist_batch, y_batch in train_loader:
            X_counts_batch, X_dist_batch, y_batch = X_counts_batch.to(device), X_dist_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_counts_batch, X_dist_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_counts_val, X_dist_val, y_val_batch in val_loader:
                X_counts_val, X_dist_val, y_val_batch = X_counts_val.to(device), X_dist_val.to(device), y_val_batch.to(device)
                outputs = model(X_counts_val, X_dist_val)
                loss = criterion(outputs, y_val_batch)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += y_val_batch.size(0)
                correct += (predicted == y_val_batch).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        # Save checkpoint
        if save_every > 0 and (epoch + 1) % save_every == 0:
            save_path = os.path.join(save_dir, f"{save_name}_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at {save_path}")

    print("Training complete.")

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

    from sklearn.metrics import accuracy_score, f1_score

    acc = accuracy_score(all_labels.numpy(), all_preds.numpy())
    f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average="macro")
    
    return acc, f1
