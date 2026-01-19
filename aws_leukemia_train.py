import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
import torch_geometric.nn
import boto3
from datetime import datetime
import json


S3_DATASET_BUCKET = 'leukemia-dataset-project'  
S3_RESULTS_BUCKET = 'leukemia-results-project'  
DATASET_FOLDER = 'Segmented'  
LOCAL_DATA_DIR = '/tmp/leukemia_data'  
LOCAL_RESULTS_DIR = '/tmp/leukemia_results'


s3_client = boto3.client('s3')


CLASS_NAMES = ['Benign', 'Early', 'Pre', 'Pro']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def download_dataset_from_s3():
    """Download dataset from S3 to local EC2 storage."""
    print("="*60)
    print("DOWNLOADING DATASET FROM S3")
    print("="*60)
    
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    
    
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_DATASET_BUCKET, Prefix=DATASET_FOLDER)
    
    file_count = 0
    for page in pages:
        if 'Contents' not in page:
            continue
        
        for obj in page['Contents']:
            s3_key = obj['Key']
            
            
            if s3_key.endswith('/'):
                continue
            
            
            local_path = os.path.join(LOCAL_DATA_DIR, s3_key)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            
            try:
                s3_client.download_file(S3_DATASET_BUCKET, s3_key, local_path)
                file_count += 1
                if file_count % 100 == 0:
                    print(f"Downloaded {file_count} files...")
            except Exception as e:
                print(f"Error downloading {s3_key}: {e}")
    
    print(f"\n✅ Successfully downloaded {file_count} files from S3")
    print("="*60 + "\n")


def upload_to_s3(local_file_path, s3_key):
    """Upload a file to S3."""
    try:
        s3_client.upload_file(local_file_path, S3_RESULTS_BUCKET, s3_key)
        print(f"✅ Uploaded to S3: s3://{S3_RESULTS_BUCKET}/{s3_key}")
        return True
    except Exception as e:
        print(f"❌ Error uploading {s3_key}: {e}")
        return False


def image_to_graph(image_path):
    """Convert an image to a graph representation."""
    try:
        image = Image.open(image_path).convert('L').resize((96, 96))
        image_array = np.array(image) / 255.0
        
        height, width = image_array.shape
        data = Data()
        data.x = torch.tensor(image_array.flatten(), dtype=torch.float).view(-1, 1)
        
        edges = []
        for i in range(height):
            for j in range(width):
                if j < width - 1:
                    edges.append((i * width + j, i * width + (j + 1)))
                if i < height - 1:
                    edges.append((i * width + j, (i + 1) * width + j))
        
        data.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return data
    except Exception as e:
        print(f"Failed to convert image to graph: {image_path}. Error: {e}")
        return None


def process_dataset(dataset_dir):
    """Process all images in the dataset directory into graph structures."""
    graph_data_list = []
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(dataset_dir, DATASET_FOLDER, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found: {class_dir}")
            continue
        
        print(f"\nProcessing class: {class_name}")
        class_label = CLASS_TO_IDX[class_name]
        
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
    """Graph Neural Network model definition."""
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
    training_history = []
    
    if class_weights is not None:
        class_weights = class_weights.to(next(model.parameters()).device)
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in data_loader:
            optimizer.zero_grad()
            output = model(batch)
            
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
        
        epoch_stats = {
            'epoch': epoch + 1,
            'loss': avg_loss,
            'accuracy': accuracy
        }
        training_history.append(epoch_stats)
        
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.2f}%')
    
    return training_history


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
    
    
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    
    cm = confusion_matrix(y_true, y_pred)
    print("="*50)
    print("CONFUSION MATRIX")
    print("="*50)
    print(cm)
    
    
    os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Leukemia Classification')
    plt.tight_layout()
    confusion_matrix_path = os.path.join(LOCAL_RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    
    accuracy = 100 * np.trace(cm) / np.sum(cm)
    print(f"\nOverall Test Accuracy: {accuracy:.2f}%")
    
    return report, cm, accuracy, confusion_matrix_path


def save_results(model, training_history, report, cm, accuracy):
    """Save all results locally and upload to S3."""
    os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    
    model_path = os.path.join(LOCAL_RESULTS_DIR, f'leukemia_gnn_model_{timestamp}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Model saved locally: {model_path}")
    
    
    history_path = os.path.join(LOCAL_RESULTS_DIR, f'training_history_{timestamp}.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    
    metrics = {
        'timestamp': timestamp,
        'test_accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    metrics_path = os.path.join(LOCAL_RESULTS_DIR, f'evaluation_metrics_{timestamp}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    
    print("\n" + "="*50)
    print("UPLOADING RESULTS TO S3")
    print("="*50)
    
    upload_to_s3(model_path, f'models/leukemia_gnn_model_{timestamp}.pth')
    upload_to_s3(history_path, f'metrics/training_history_{timestamp}.json')
    upload_to_s3(metrics_path, f'metrics/evaluation_metrics_{timestamp}.json')
    upload_to_s3(os.path.join(LOCAL_RESULTS_DIR, 'confusion_matrix.png'), 
                 f'visualizations/confusion_matrix_{timestamp}.png')
    
    print("\n✅ All results uploaded to S3!")
    print(f"S3 Bucket: s3://{S3_RESULTS_BUCKET}/")


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("AWS-INTEGRATED LEUKEMIA DETECTION WITH GNN")
    print("="*60 + "\n")
    
    
    download_dataset_from_s3()
    
    
    graph_data_list = process_dataset(LOCAL_DATA_DIR)
    
    if not graph_data_list or len(graph_data_list) == 0:
        print("❌ ERROR: No graph data created. Exiting.")
        return
    
    print(f"✅ Dataset loaded successfully with {len(graph_data_list)} samples")
    
    
    class_counts = {i: 0 for i in range(4)}
    for data in graph_data_list:
        class_counts[data.y.item()] += 1
    
    print("\n📊 Class Distribution:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {class_counts[i]} samples")
    
    
    total_samples = len(graph_data_list)
    class_weights = torch.tensor([total_samples / (4 * class_counts[i]) if class_counts[i] > 0 else 1.0 
                                  for i in range(4)], dtype=torch.float)
    print(f"\n⚖️  Class Weights: {class_weights.numpy()}")
    
    
    train_data, test_data = train_test_split(graph_data_list, test_size=0.2, random_state=42, 
                                             stratify=[d.y.item() for d in graph_data_list])
    print(f"\n📂 Training samples: {len(train_data)}")
    print(f"📂 Testing samples: {len(test_data)}")
    
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    
    
    model = GNN(num_classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\n" + "="*60)
    print("🚀 TRAINING STARTED - 25 EPOCHS (96x96 images)")
    print("="*60 + "\n")
    
    
    training_history = train_model(model, train_loader, optimizer, epochs=25, class_weights=class_weights)
    
    print("\n" + "="*60)
    print("📊 EVALUATION ON TEST SET")
    print("="*60 + "\n")
    
    
    report, cm, accuracy, cm_path = evaluate_model(model, test_loader)
    
    
    save_results(model, training_history, report, cm, accuracy)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📁 Check your S3 bucket: s3://{S3_RESULTS_BUCKET}/")
    print("   - models/ (trained model)")
    print("   - metrics/ (training history & evaluation)")
    print("   - visualizations/ (confusion matrix)")


if __name__ == '__main__':
    main()