
import torch
from sklearn.metrics import accuracy_score, f1_score

def evaluate_on_query(model, X_query_tensor, y_query, label_encoder):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(X_query_tensor)
        _, predicted = torch.max(outputs.data, 1)
    
    # Convert numeric labels back to original cell types
    predicted_celltypes = label_encoder.inverse_transform(predicted.cpu().numpy())
    true_celltypes = label_encoder.inverse_transform(y_query)
    
    # Calculate accuracy and F1 score
    accuracy = accuracy_score(true_celltypes, predicted_celltypes)
    f1 = f1_score(true_celltypes, predicted_celltypes, average='macro')
    
    return predicted_celltypes, accuracy, f1

