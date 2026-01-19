Graph Neural Network-Based Leukemia Detection System on AWS
An end-to-end cloud-based medical imaging system leveraging Graph Neural Networks for automated classification of microscopic blood cell images into four leukemia stages: Benign, Early, Pre, and Pro. Deployed on AWS with real-time prediction capabilities.
Overview
This project demonstrates the application of Graph Neural Networks to medical image analysis. Unlike traditional CNNs, this approach represents images as graphs, capturing spatial relationships through spectral convolutions. The system achieves 84% test accuracy with 1-2 second inference latency.
Key Features

<img width="1224" height="786" alt="image" src="https://github.com/user-attachments/assets/741784c9-5112-446c-a67e-6b83d99d251d" />


Graph Neural Network: 3-layer GCN architecture transforming 96×96 images into 9,216-node graphs
AWS Deployment: Complete cloud infrastructure using EC2, S3, IAM, and CloudWatch
Real-time API: Flask REST API with secure IAM authentication
Web Interface: Responsive frontend with drag-and-drop image upload
High Performance: 84% accuracy on 4-class classification with sub-2-second predictions

Architecture
Model Architecture:

Input: 96×96 grayscale microscopic blood cell images
Graph representation: 9,216 nodes, ~18,432 edges (4-connected grid)
GCN layers: 1→16→32→64 feature dimensions
Global mean pooling followed by fully connected output layer
Output: Probability distribution over 4 classes

AWS Infrastructure:

EC2 t2.micro instance for model training and Flask API hosting
S3 buckets for dataset storage, model artifacts, user uploads, and static website hosting
IAM role-based authentication for secure EC2-to-S3 communication
CloudWatch for monitoring metrics and application logs

Dataset
Source: Kaggle ALL (Acute Lymphoblastic Leukemia) Dataset
Details:

Total images: ~650 microscopic blood cell images
Classes: Benign, Early, Pre, Pro
Format: JPG, PNG, BMP
Split: 80% training, 20% testing (stratified)

Model Performance
MetricValueOverall Accuracy84%Benign Recall58%Early Recall81%Pre Recall87%Pro Recall98%Inference Time1-2 seconds
Installation
Prerequisites

Python 3.9+
AWS Account
AWS CLI configured

Local Setup
bashgit clone https://github.com/yourusername/gnn-leukemia-detection.git
cd gnn-leukemia-detection
pip install -r requirements.txt
AWS Deployment
1. Create S3 Buckets:
bashaws s3 mb s3://leukemia-dataset-project
aws s3 mb s3://leukemia-results-project
aws s3 mb s3://leukemia-user-uploads
aws s3 mb s3://leukemia-detection-ui
2. Upload Dataset:
bashaws s3 sync ./dataset/Segmented s3://leukemia-dataset-project/Segmented/
3. Launch EC2 Instance:

Instance type: t2.micro
AMI: Amazon Linux 2023
Security group: Allow SSH (22) and HTTP (5000)
IAM role: LeukemiaEC2Role with S3 and CloudWatch access

4. Install Dependencies on EC2:
bashsudo yum update -y
sudo yum install python3.9 python3.9-pip -y
pip3.9 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3.9 install torch-geometric torch-scatter torch-sparse
pip3.9 install flask flask-cors boto3 pillow numpy scikit-learn matplotlib seaborn
5. Train Model:
bashpython3.9 aws_leukemia_train.py
6. Start API Server:
bashpython3.9 flask_api.py
7. Deploy Frontend:
bashaws s3 cp index.html s3://leukemia-detection-ui/
aws s3 website s3://leukemia-detection-ui/ --index-document index.html
Usage
Training
bashpython3.9 aws_leukemia_train.py
Training configuration:

Optimizer: Adam (lr=0.001)
Batch size: 16
Epochs: 25
Loss: Weighted Negative Log-Likelihood
Training time: ~45 minutes on t2.micro

Inference via API
Health Check:
bashcurl http://your-ec2-ip:5000/health
Prediction:
bashcurl -X POST http://your-ec2-ip:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image", "filename": "cell.jpg"}'
Response:
json{
  "success": true,
  "prediction": {
    "predicted_class": "Early",
    "confidence": 95.1,
    "all_probabilities": {
      "Benign": 3.2,
      "Early": 95.1,
      "Pre": 1.1,
      "Pro": 0.7
    }
  }
}
Web Interface
Access the web interface at: http://leukemia-detection-ui.s3-website-us-east-1.amazonaws.com
Upload a microscopic blood cell image via drag-and-drop or file selection to receive instant classification results.
API Documentation
Endpoints
GET /health

Description: Health check endpoint
Response: {"status": "healthy", "message": "Leukemia Detection API is running"}

POST /predict

Description: Classify blood cell image
Request Body:

json  {
    "image": "base64_encoded_image_string",
    "filename": "image_name.jpg"
  }

Response: Prediction results with class probabilities

Project Structure
gnn-leukemia-detection/
├── aws_leukemia_train.py    # Training script
├── flask_api.py              # Flask REST API
├── index.html                # Frontend UI
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── LICENSE                   # MIT License
Technologies Used
Machine Learning:

PyTorch 2.0.1
PyTorch Geometric
scikit-learn
NumPy

Cloud & Infrastructure:

AWS EC2 (t2.micro)
AWS S3
AWS IAM
AWS CloudWatch

Backend:

Flask 3.0.0
Flask-CORS
Boto3 (AWS SDK)

Frontend:

HTML5
CSS3 (Tailwind)
JavaScript (ES6)

AWS Services Configuration
IAM Role: LeukemiaEC2Role

Policies: AmazonS3FullAccess, CloudWatchLogsFullAccess
Trust relationship: ec2.amazonaws.com

S3 Buckets:

leukemia-dataset-project: Training dataset storage
leukemia-results-project: Model artifacts and metrics
leukemia-user-uploads: User-uploaded images and predictions
leukemia-detection-ui: Static website hosting

Security Group:

Inbound: SSH (22), Custom TCP (5000)
Outbound: All traffic

Cost Optimization
This project is designed to run entirely within AWS Free Tier:

EC2: 750 hours/month of t2.micro
S3: 5GB storage, 20K GET requests, 2K PUT requests
CloudWatch: 10 custom metrics, 5GB logs
Estimated monthly cost: $0 (within free tier limits)

Future Enhancements

Implement API Gateway for better scalability
Add user authentication and session management
Deploy model to AWS SageMaker for managed inference
Implement A/B testing for model versions
Add batch processing capabilities
Integrate with DICOM medical imaging standards
Implement explainability features (GNNExplainer, GradCAM)

Contributing
Contributions are welcome! Please follow these steps:

Fork the repository
Create a feature branch (git checkout -b feature/YourFeature)
Commit changes (git commit -m 'Add YourFeature')
Push to branch (git push origin feature/YourFeature)
Open a Pull Request

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Dataset: Kaggle ALL (Acute Lymphoblastic Leukemia) Dataset
PyTorch Geometric library for graph neural network implementations
AWS Free Tier for cloud infrastructure
Flask framework for API development
