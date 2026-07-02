import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
from config import OUTPUT_DIR, PROCESSED_DATASET, interim_csb_path

PROCESSED_PATH = PROCESSED_DATASET
CSB_PATH = interim_csb_path("20172024")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Geometric Graph Convolution Layer
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        # input: (batch_size, num_nodes, in_features)
        # adj: (batch_size, num_nodes, num_nodes) adjacency matrix
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        return self.dropout(output)

# Memory-Efficient Geometric CNN for Spatial Crop Classification  
class GeometricCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super(GeometricCNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Store dimensions for dynamic sizing
        self.spatial_dim = 5  # Will be updated dynamically based on available features
        
        # Spatial feature extractor (focuses on geometric + soil features)
        self.spatial_conv = nn.Sequential(
            nn.Linear(self.spatial_dim, hidden_size // 2),  # Spatial + soil features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 2)
        )
        
        # Temporal feature extractor  
        self.temporal_conv = nn.Sequential(
            nn.Linear(input_size - self.spatial_dim, hidden_size // 2),  # Temporal features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 2)
        )
        
        # Spatial-temporal fusion layers (2-layer CNN)
        self.fusion_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x, spatial_dim=None):
        # Update spatial dimension if provided
        if spatial_dim is not None:
            self.spatial_dim = spatial_dim
        
        # Split features into spatial and temporal components
        # Last features are spatial (geometric + soil), first are temporal
        spatial_features = x[:, -self.spatial_dim:]
        temporal_features = x[:, :-self.spatial_dim]
        
        # Extract spatial and temporal representations
        spatial_rep = self.spatial_conv(spatial_features)
        temporal_rep = self.temporal_conv(temporal_features)
        
        # Fuse spatial and temporal information
        combined = torch.cat([spatial_rep, temporal_rep], dim=1)
        fused_features = self.fusion_layers(combined)
        
        # Classification
        output = self.classifier(fused_features)
        return output

# Geometric RNN for Temporal-Spatial Crop Classification
class GeometricRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super(GeometricRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # Store dimensions for dynamic sizing
        self.spatial_dim = 5  # Will be updated dynamically
        
        # Spatial feature encoder (geometric + soil features)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(self.spatial_dim, hidden_size // 2),  # Spatial + soil features
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temporal feature encoder  
        self.temporal_encoder = nn.Sequential(
            nn.Linear(input_size - self.spatial_dim, hidden_size // 2),  # Temporal features
            nn.ReLU(), 
            nn.Dropout(dropout)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x, spatial_dim=None):
        # Update spatial dimension if provided
        if spatial_dim is not None:
            self.spatial_dim = spatial_dim
            
        batch_size, seq_len, features = x.shape
        
        # Split features into spatial and temporal components  
        # Last features are spatial (geometric + soil), first are temporal
        spatial_features = x[:, :, -self.spatial_dim:]
        temporal_features = x[:, :, :-self.spatial_dim]
        
        # Encode spatial and temporal features separately
        spatial_encoded = self.spatial_encoder(spatial_features)
        temporal_encoded = self.temporal_encoder(temporal_features)
        
        # Combine spatial and temporal encodings
        combined_features = torch.cat([temporal_encoded, spatial_encoded], dim=-1)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(combined_features)
        
        # Use last time step output for classification
        final_output = lstm_out[:, -1, :]
        
        # Classification
        output = self.classifier(final_output)
        return output

# Data loading and preprocessing functions
def load_and_preprocess_data(sample_size=100000):
    """Load and preprocess data for geometric deep learning"""
    print("Loading data...")
    df = pd.read_parquet(PROCESSED_PATH)
    
    # Sample for computational efficiency
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} records")
    
    # Define crop mapping
    crop_mapping = {
        'Corn': 0, 'Soybeans': 1, 'Alfalfa': 2, 
        'Combined Hay/Grass': 3, 'Other': 4
    }
    
    # Convert targets to numeric
    df['Target'] = df['Crop_Type'].map(crop_mapping)
    
    # Define feature sets
    geometric_features = [
        'Longitude_Norm', 'Latitude_Norm', 'County_Crop_Diversity',
        'County_Avg_Field_Size', 'CNTYFIPS'
    ]
    
    # Soil features - important for crop prediction
    soil_features = [
        'slopegradwta',        # slope gradient (%)
        'aws050wta',           # available water storage 0-50 cm  
        'aws0100wta',          # available water storage 0-100 cm
        'drainage_numeric'     # ordinal drainage class
    ]
    
    # Handle categorical soil features (convert to numeric if present)
    categorical_soil = ['hydro_group_primary']  # hydrologic group (A, B, C, D)
    
    temporal_features = [
        'Planting_Precip', 'Growing_GDD', 'Planting_Precip_Lag1',
        'Growing_GDD_Lag1', 'Crop_Lag1', 'Crop_Lag2'
    ]
    
    # Check feature availability
    available_geometric = [f for f in geometric_features if f in df.columns]
    available_temporal = [f for f in temporal_features if f in df.columns]
    available_soil = [f for f in soil_features if f in df.columns and df[f].notna().any()]
    
    # Handle categorical soil features - encode them as numeric
    available_categorical_soil = []
    for cat_feature in categorical_soil:
        if cat_feature in df.columns and df[cat_feature].notna().any():
            # Convert categorical to numeric (e.g., A=0, B=1, C=2, D=3)
            if cat_feature == 'hydro_group_primary':
                hydro_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
                df[f'{cat_feature}_numeric'] = df[cat_feature].map(hydro_mapping).fillna(-1)
                available_categorical_soil.append(f'{cat_feature}_numeric')
    
    print(f"Available geometric features: {available_geometric}")
    print(f"Available temporal features: {available_temporal}")
    print(f"Available soil features: {available_soil}")
    print(f"Available categorical soil features: {available_categorical_soil}")
    
    all_features = available_temporal + available_soil + available_categorical_soil + available_geometric
    
    # Create feature matrix
    X = df[all_features].fillna(0).values
    y = df['Target'].values
    
    # Get coordinates for spatial features (if available)
    if 'Longitude_Norm' in df.columns and 'Latitude_Norm' in df.columns:
        coordinates = df[['Longitude_Norm', 'Latitude_Norm']].fillna(0).values
    else:
        coordinates = None
        print("Warning: No coordinate features found")
    
    return X, y, coordinates, all_features, available_temporal, available_soil, available_categorical_soil, available_geometric

def build_spatial_adjacency(coordinates, k=5):
    """Build spatial adjacency matrix using k-nearest neighbors"""
    if coordinates is None:
        return None
    
    # Find k-nearest neighbors based on spatial coordinates
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coordinates)
    distances, indices = nbrs.kneighbors(coordinates)
    
    n_samples = len(coordinates)
    adj_matrix = np.zeros((n_samples, n_samples))
    
    # Build adjacency matrix
    for i in range(n_samples):
        for j in range(1, k+1):  # Skip self (index 0)
            neighbor_idx = indices[i, j]
            # Use inverse distance as edge weight
            distance = distances[i, j]
            if distance > 0:
                weight = 1.0 / (1.0 + distance)
                adj_matrix[i, neighbor_idx] = weight
    
    # Normalize adjacency matrix
    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    adj_matrix = adj_matrix / row_sums
    
    return adj_matrix

def create_temporal_sequences(X, y, coordinates, sequence_length=3):
    """Create temporal sequences for RNN training"""
    # For simplicity, create sequences by sliding window over samples
    # In practice, this should be done per field over time
    sequences_X = []
    sequences_y = []
    
    for i in range(len(X) - sequence_length + 1):
        seq_x = X[i:i+sequence_length]
        seq_y = y[i+sequence_length-1]  # Predict last item in sequence
        sequences_X.append(seq_x)
        sequences_y.append(seq_y)
    
    return np.array(sequences_X), np.array(sequences_y)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, 
                gradient_clip=1.0, model_type="CNN"):
    """Training loop for geometric models"""
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_data in train_loader:
            features, labels = batch_data
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                features, labels = batch_data
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                num_val_batches += 1
        
        avg_val_loss = val_loss / num_val_batches
        val_accuracy = 100 * correct / total
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, '
                  f'Val Acc: {val_accuracy:.2f}%')
    
    return train_losses, val_losses, val_accuracies

def evaluate_model(model, test_loader, model_type="CNN"):
    """Evaluate model performance"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            features, labels = batch_data
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Crop names for better interpretation
    crop_names = ['Corn', 'Soybeans', 'Alfalfa', 'Combined Hay/Grass', 'Other']
    report = classification_report(all_labels, all_predictions, 
                                 target_names=crop_names, digits=4)
    
    return accuracy, report, all_predictions, all_labels

def plot_training_curves(train_losses, val_losses, val_accuracies, model_name):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(val_accuracies, label='Val Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{model_name} - Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filepath = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_training_curves.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved training curves: {filepath}")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix"""
    crop_names = ['Corn', 'Soybeans', 'Alfalfa', 'Combined Hay/Grass', 'Other']
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=crop_names, yticklabels=crop_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted Crop Type')
    plt.ylabel('True Crop Type')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    filepath = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_confusion_matrix.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix: {filepath}")
    plt.show()

# Main training function
def main():
    """Main function to train Geometric CNN and RNN models"""
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Model hyperparameters (adjustable in notebook)
    hyperparameters = {
        'input_size': 10,        # Will be adjusted based on actual features
        'hidden_size': 32,
        'learning_rate': 0.0001,
        'num_epochs': 500,
        'dropout': 0.2,
        'batch_size': 32,
        'gradient_clip': 1.0,
        'sample_size': 50000,
        'k_neighbors': 5,
        'sequence_length': 3
    }
    
    print("="*60)
    print("GEOMETRIC DEEP LEARNING FOR CROP CLASSIFICATION")
    print("="*60)
    print(f"Hyperparameters: {hyperparameters}")
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    X, y, coordinates, feature_names, available_temporal, available_soil, available_categorical_soil, available_geometric = load_and_preprocess_data(
        sample_size=hyperparameters['sample_size']
    )
    
    # Update input size based on actual features
    hyperparameters['input_size'] = X.shape[1]
    num_classes = len(np.unique(y))
    
    # Calculate spatial dimension (soil + categorical soil + geometric features)
    spatial_dim = len(available_soil + available_categorical_soil + available_geometric)
    
    print(f"Data shape: {X.shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Feature names: {feature_names}")
    print(f"Spatial features (soil + geometric): {spatial_dim}")
    print(f"Temporal features: {len(available_temporal)}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Calculate class weights for balancing
    class_counts = np.bincount(y)
    total_samples = len(y)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train size: {len(X_train_scaled)}")
    print(f"Validation size: {len(X_val_scaled)}")
    print(f"Test size: {len(X_test_scaled)}")
    
    # Prepare data for Memory-Efficient CNN
    print("\n2. Preparing Memory-Efficient Geometric CNN data...")
    
    # Create CNN datasets (no adjacency matrices needed)
    cnn_train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.LongTensor(y_train)
    )
    
    cnn_val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.LongTensor(y_val)
    )
    
    cnn_test_dataset = TensorDataset(
        torch.FloatTensor(X_test_scaled),
        torch.LongTensor(y_test)
    )
    
    cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
    cnn_val_loader = DataLoader(cnn_val_dataset, batch_size=hyperparameters['batch_size'])
    cnn_test_loader = DataLoader(cnn_test_dataset, batch_size=hyperparameters['batch_size'])
    
    print("Memory-efficient CNN datasets prepared")
    
    # Prepare data for RNN (requires temporal sequences)
    print("\n3. Preparing Geometric RNN data...")
    X_seq_train, y_seq_train = create_temporal_sequences(
        X_train_scaled, y_train, None, hyperparameters['sequence_length']
    )
    X_seq_val, y_seq_val = create_temporal_sequences(
        X_val_scaled, y_val, None, hyperparameters['sequence_length']
    )
    X_seq_test, y_seq_test = create_temporal_sequences(
        X_test_scaled, y_test, None, hyperparameters['sequence_length']
    )
    
    # Create RNN datasets
    rnn_train_dataset = TensorDataset(
        torch.FloatTensor(X_seq_train),
        torch.LongTensor(y_seq_train)
    )
    
    rnn_val_dataset = TensorDataset(
        torch.FloatTensor(X_seq_val),
        torch.LongTensor(y_seq_val)
    )
    
    rnn_test_dataset = TensorDataset(
        torch.FloatTensor(X_seq_test),
        torch.LongTensor(y_seq_test)
    )
    
    rnn_train_loader = DataLoader(rnn_train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
    rnn_val_loader = DataLoader(rnn_val_dataset, batch_size=hyperparameters['batch_size'])
    rnn_test_loader = DataLoader(rnn_test_dataset, batch_size=hyperparameters['batch_size'])
    
    print(f"RNN sequences prepared - shape: {X_seq_train.shape}")
    
    # Initialize models
    results = {}
    
    # Train Geometric CNN
    if cnn_train_loader is not None:
        print("\n4. Training Memory-Efficient Geometric CNN...")
        cnn_model = GeometricCNN(
            input_size=hyperparameters['input_size'],
            hidden_size=hyperparameters['hidden_size'],
            output_size=num_classes,
            dropout=hyperparameters['dropout']
        ).to(device)
        
        cnn_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=hyperparameters['learning_rate'])
        
        # Update model with correct spatial dimension
        cnn_model.spatial_dim = spatial_dim
        # Rebuild the spatial conv layer with correct input size
        cnn_model.spatial_conv = nn.Sequential(
            nn.Linear(spatial_dim, hyperparameters['hidden_size'] // 2),
            nn.ReLU(),
            nn.Dropout(hyperparameters['dropout']),
            nn.Linear(hyperparameters['hidden_size'] // 2, hyperparameters['hidden_size'] // 2)
        ).to(device)
        # Rebuild temporal conv layer  
        cnn_model.temporal_conv = nn.Sequential(
            nn.Linear(hyperparameters['input_size'] - spatial_dim, hyperparameters['hidden_size'] // 2),
            nn.ReLU(),
            nn.Dropout(hyperparameters['dropout']),
            nn.Linear(hyperparameters['hidden_size'] // 2, hyperparameters['hidden_size'] // 2)
        ).to(device)
        
        # Train CNN
        cnn_train_losses, cnn_val_losses, cnn_val_accs = train_model(
            cnn_model, cnn_train_loader, cnn_val_loader, 
            cnn_criterion, cnn_optimizer, hyperparameters['num_epochs'],
            hyperparameters['gradient_clip'], "CNN"
        )
        
        # Evaluate CNN
        cnn_accuracy, cnn_report, cnn_preds, cnn_labels = evaluate_model(
            cnn_model, cnn_test_loader, "CNN"
        )
        
        results['CNN'] = {
            'model': cnn_model,
            'accuracy': cnn_accuracy,
            'report': cnn_report,
            'predictions': cnn_preds,
            'labels': cnn_labels,
            'train_losses': cnn_train_losses,
            'val_losses': cnn_val_losses, 
            'val_accuracies': cnn_val_accs
        }
        
        print(f"\nCNN Results:")
        print(f"Test Accuracy: {cnn_accuracy:.4f}")
        print(f"Classification Report:\n{cnn_report}")
        
        # Plot CNN results
        plot_training_curves(cnn_train_losses, cnn_val_losses, cnn_val_accs, "Memory_Efficient_CNN")
        plot_confusion_matrix(cnn_labels, cnn_preds, "Memory_Efficient_CNN")
    
    # Train Geometric RNN
    print("\n5. Training Geometric RNN...")
    rnn_model = GeometricRNN(
        input_size=hyperparameters['input_size'],
        hidden_size=hyperparameters['hidden_size'],
        output_size=num_classes,
        num_layers=2,
        dropout=hyperparameters['dropout']
    ).to(device)
    
    rnn_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=hyperparameters['learning_rate'])
    
    # Update RNN model with correct spatial dimension
    rnn_model.spatial_dim = spatial_dim
    # Rebuild spatial encoder with correct input size
    rnn_model.spatial_encoder = nn.Sequential(
        nn.Linear(spatial_dim, hyperparameters['hidden_size'] // 2),
        nn.ReLU(),
        nn.Dropout(hyperparameters['dropout'])
    ).to(device)
    # Rebuild temporal encoder
    rnn_model.temporal_encoder = nn.Sequential(
        nn.Linear(hyperparameters['input_size'] - spatial_dim, hyperparameters['hidden_size'] // 2),
        nn.ReLU(),
        nn.Dropout(hyperparameters['dropout'])
    ).to(device)
    
    # Train RNN
    rnn_train_losses, rnn_val_losses, rnn_val_accs = train_model(
        rnn_model, rnn_train_loader, rnn_val_loader,
        rnn_criterion, rnn_optimizer, hyperparameters['num_epochs'],
        hyperparameters['gradient_clip'], "RNN"
    )
    
    # Evaluate RNN
    rnn_accuracy, rnn_report, rnn_preds, rnn_labels = evaluate_model(
        rnn_model, rnn_test_loader, "RNN"
    )
    
    results['RNN'] = {
        'model': rnn_model,
        'accuracy': rnn_accuracy,
        'report': rnn_report,
        'predictions': rnn_preds,
        'labels': rnn_labels,
        'train_losses': rnn_train_losses,
        'val_losses': rnn_val_losses,
        'val_accuracies': rnn_val_accs
    }
    
    print(f"\nRNN Results:")
    print(f"Test Accuracy: {rnn_accuracy:.4f}")
    print(f"Classification Report:\n{rnn_report}")
    
    # Plot RNN results
    plot_training_curves(rnn_train_losses, rnn_val_losses, rnn_val_accs, "Geometric_RNN")
    plot_confusion_matrix(rnn_labels, rnn_preds, "Geometric_RNN")
    
    # Model comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'Model': f'Geometric_{model_name}',
            'Test_Accuracy': f"{result['accuracy']:.4f}",
            'Best_Val_Accuracy': f"{max(result['val_accuracies']):.2f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, "geometric_models_comparison.csv"), index=False)
    print(f"\nSaved model comparison to: {os.path.join(OUTPUT_DIR, 'geometric_models_comparison.csv')}")
    
    # Save model checkpoints
    if 'CNN' in results:
        torch.save(results['CNN']['model'].state_dict(), 
                  os.path.join(OUTPUT_DIR, "geometric_cnn_model.pth"))
        print("Saved CNN model checkpoint")
    
    torch.save(results['RNN']['model'].state_dict(),
              os.path.join(OUTPUT_DIR, "geometric_rnn_model.pth"))
    print("Saved RNN model checkpoint")
    
    print("\nGeometric Deep Learning training completed!")
    return results

if __name__ == "__main__":
    results = main()

