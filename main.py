import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch_geometric.nn

# Directory containing the dataset (using raw string to handle backslashes)
dataset_dir = r'C:\Users\visha\OneDrive\ドキュメント\Cloud Computing Project\Segmented'

# Class mapping
CLASS_NAMES = ['Benign', 'Early', 'Pre', 'Pro']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def image_to_graph(image_path):
    """Convert an image to a graph representation."""
    try:
        # Load and preprocess the image (increased size for better accuracy)
        image = Image.open(image_path).convert('L').resize((96, 96))
        image_array = np.array(image) / 255.0  # Normalize pixel values
        
        height, width = image_array.shape
        data = Data()
        data.x = torch.tensor(image_array.flatten(), dtype=torch.float).view(-1, 1)
        
        # Create edges for the graph (grid structure)
        edges = []
        for i in range(height):
            for j in range(width):
                if j < width - 1:
                    edges.append((i * width + j, i * width + (j + 1)))  # Right
                if i < height - 1:
                    edges.append((i * width + j, (i + 1) * width + j))  # Down
        
        data.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return data
    except Exception as e:
        print(f"Failed to convert image to graph data: {image_path}. Error: {e}")
        return None


def process_dataset(dataset_dir):
    """Process all images in the dataset directory into graph structures."""
    graph_data_list = []
    
    # Iterate through each class folder
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(dataset_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found: {class_dir}")
            continue
        
        print(f"\nProcessing class: {class_name}")
        class_label = CLASS_TO_IDX[class_name]
        
        # Process all images in the class folder
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"Found {len(image_files)} images in {class_name}")
        
        for img_file in image_files:
            image_path = os.path.join(class_dir, img_file)
            graph_data = image_to_graph(image_path)
            
            if graph_data is not None:
                graph_data.y = torch.tensor([class_label], dtype=torch.long)
                graph_data_list.append(graph_data)
        
        print(f"Successfully processed {len([g for g in graph_data_list if g.y.item() == class_label])} images from {class_name}")
    
    print(f"\n{'='*50}")
    print(f"Total processed graphs: {len(graph_data_list)}")
    print(f"{'='*50}\n")
    return graph_data_list


class GNN(torch.nn.Module):
    """Graph Neural Network model definition for 4-class classification."""
    def __init__(self, num_classes=4):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 64)
        self.fc = torch.nn.Linear(64, num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = torch_geometric.nn.global_mean_pool(x, data.batch)
        return F.log_softmax(self.fc(x), dim=1)


def train_model(model, data_loader, optimizer, epochs=25, class_weights=None):
    """Train the GNN model."""
    model.train()
    
    # Move class weights to appropriate device if provided
    if class_weights is not None:
        class_weights = class_weights.to(next(model.parameters()).device)
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in data_loader:
            optimizer.zero_grad()
            output = model(batch)
            
            # Use weighted loss if class weights provided
            if class_weights is not None:
                loss = F.nll_loss(output, batch.y, weight=class_weights)
            else:
                loss = F.nll_loss(output, batch.y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += batch.y.size(0)
            correct += (predicted == batch.y).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(data_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.2f}%')


def evaluate_model(model, dataloader):
    """Evaluate the trained model."""
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for data in dataloader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Classification Report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("="*50)
    print("CONFUSION MATRIX")
    print("="*50)
    print(cm)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Leukemia Classification')
    plt.tight_layout()
    plt.show()
    
    # Calculate and display accuracy
    accuracy = 100 * np.trace(cm) / np.sum(cm)
    print(f"\nOverall Test Accuracy: {accuracy:.2f}%")


# Main script execution
if __name__ == '__main__':
    print("Starting Leukemia Detection with GNN")
    print("="*50)
    
    # Process dataset
    graph_data_list = process_dataset(dataset_dir)
    
    if graph_data_list and len(graph_data_list) > 0:
        print(f"Dataset loaded successfully with {len(graph_data_list)} samples")
        
        # Calculate class distribution
        class_counts = {i: 0 for i in range(4)}
        for data in graph_data_list:
            class_counts[data.y.item()] += 1
        
        print("\nClass Distribution:")
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {name}: {class_counts[i]} samples")
        
        # Calculate class weights (inverse frequency)
        total_samples = len(graph_data_list)
        class_weights = torch.tensor([total_samples / (4 * class_counts[i]) if class_counts[i] > 0 else 1.0 
                                      for i in range(4)], dtype=torch.float)
        print(f"\nClass Weights (to handle imbalance): {class_weights.numpy()}")
        
        # Split data
        train_data, test_data = train_test_split(graph_data_list, test_size=0.2, random_state=42, 
                                                 stratify=[d.y.item() for d in graph_data_list])
        print(f"\nTraining samples: {len(train_data)}")
        print(f"Testing samples: {len(test_data)}")
        
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
        
        # Initialize model
        model = GNN(num_classes=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        print("\n" + "="*50)
        print("TRAINING STARTED - 25 EPOCHS (96x96 images)")
        print("="*50 + "\n")
        
        # Train model
        train_model(model, train_loader, optimizer, epochs=25, class_weights=class_weights)
        
        print("\n" + "="*50)
        print("EVALUATION ON TEST SET")
        print("="*50 + "\n")
        
        # Evaluate model
        evaluate_model(model, test_loader)
        
        # Save model
        torch.save(model.state_dict(), 'leukemia_gnn_model.pth')
        print("\nModel saved as 'leukemia_gnn_model.pth'")
    else:
        print("ERROR: No graph data was created. Please check:")
        print("1. Dataset directory path is correct")
        print("2. Folders 'Benign', 'Early', 'Pre', 'Pro' exist")
        print("3. Image files exist in these folders")