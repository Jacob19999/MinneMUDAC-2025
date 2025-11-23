"""
Advanced ML Training Pipeline with Feature Engineering, Ensemble Methods, and LSTM
Predicts Match Length using time series data with advanced techniques.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.sparse import issparse
from scipy import stats
import xgboost as xgb
import time
import warnings
import pickle
import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration class for training parameters."""
    input_file: str = "MUDAC/External Data/ML Training Test Dataset.xlsx"
    output_dir: str = "ML Pipeline/outputs"
    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.2
    target_column: str = "Match Length"
    sequence_id_column: str = "Match ID 18Char"
    date_column: str = "Completion Date"
    use_gpu: bool = True
    n_splits: int = 5  # For cross-validation
    
    # LSTM hyperparameters
    lstm_hidden_sizes: List[int] = [64, 128, 256]
    lstm_num_layers: List[int] = [2, 3]
    lstm_dropout: List[float] = [0.2, 0.3]
    lstm_learning_rate: float = 0.001
    lstm_batch_size: int = 64
    lstm_epochs: int = 100
    lstm_patience: int = 15
    
    # Ensemble parameters
    n_estimators_rf: int = 200
    n_estimators_gb: int = 200
    n_estimators_xgb: int = 200
    max_depth: int = 10
    
    # Feature engineering
    create_interactions: bool = True
    create_polynomial_features: bool = True
    polynomial_degree: int = 2
    create_time_features: bool = True
    create_aggregations: bool = True

config = Config()

# ============================================================================
# DEVICE SETUP
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")
print(f"Using device: {device}")

# ============================================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================================

class AdvancedFeatureEngineer:
    """Advanced feature engineering with interactions, polynomials, and aggregations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.polynomial_transformer = None
        self.feature_names = []
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from date columns."""
        if not self.config.create_time_features:
            return df
        
        df = df.copy()
        
        # Convert date columns
        if self.config.date_column in df.columns:
            df[self.config.date_column] = pd.to_datetime(df[self.config.date_column], errors='coerce')
            
            # Extract temporal features
            df['year'] = df[self.config.date_column].dt.year
            df['month'] = df[self.config.date_column].dt.month
            df['day'] = df[self.config.date_column].dt.day
            df['day_of_week'] = df[self.config.date_column].dt.dayofweek
            df['day_of_year'] = df[self.config.date_column].dt.dayofyear
            df['quarter'] = df[self.config.date_column].dt.quarter
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Cyclical encoding
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
            df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        
        # Match activation date features
        if 'Match Activation Date' in df.columns:
            df['Match Activation Date'] = pd.to_datetime(df['Match Activation Date'], errors='coerce')
            if self.config.date_column in df.columns:
                df['days_since_activation'] = (df[self.config.date_column] - df['Match Activation Date']).dt.days
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame, numerical_cols: List[str]) -> pd.DataFrame:
        """Create interaction features between important numerical columns."""
        if not self.config.create_interactions:
            return df
        
        df = df.copy()
        
        # Key interactions
        interaction_pairs = [
            ('Little Age', 'Big Age'),
            ('green_flag_count', 'red_flag_count'),
            ('Little Mean Household Income', 'Big Mean Household Income'),
        ]
        
        for col1, col2 in interaction_pairs:
            if col1 in df.columns and col2 in df.columns:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-6)  # Avoid division by zero
                df[f'{col1}_diff_{col2}'] = df[col1] - df[col2]
        
        return df
    
    def create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregation features grouped by Match ID."""
        if not self.config.create_aggregations or self.config.sequence_id_column not in df.columns:
            return df
        
        df = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and ID columns
        if self.config.target_column in numerical_cols:
            numerical_cols.remove(self.config.target_column)
        if self.config.sequence_id_column in numerical_cols:
            numerical_cols.remove(self.config.sequence_id_column)
        
        # Group by Match ID and create aggregations
        grouped = df.groupby(self.config.sequence_id_column)
        
        agg_features = {}
        for col in numerical_cols:
            if col in df.columns:
                agg_features[f'{col}_mean'] = grouped[col].transform('mean')
                agg_features[f'{col}_std'] = grouped[col].transform('std')
                agg_features[f'{col}_min'] = grouped[col].transform('min')
                agg_features[f'{col}_max'] = grouped[col].transform('max')
                agg_features[f'{col}_range'] = agg_features[f'{col}_max'] - agg_features[f'{col}_min']
        
        # Add to dataframe
        for name, values in agg_features.items():
            df[name] = values
        
        # Sequence position features
        df['sequence_position'] = grouped.cumcount()
        df['sequence_length'] = grouped.size().reindex(df[self.config.sequence_id_column]).values
        
        return df
    
    def create_polynomial_features(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Create polynomial features for numerical columns."""
        if not self.config.create_polynomial_features:
            return X, feature_names
        
        # Only apply to numerical features (first N features before one-hot)
        numerical_indices = [i for i, name in enumerate(feature_names) 
                            if not any(char in name for char in ['_', 'x0_', 'x1_'])]  # Rough heuristic
        
        if len(numerical_indices) == 0:
            return X, feature_names
        
        # Limit to top numerical features to avoid explosion
        top_n = min(20, len(numerical_indices))
        selected_indices = numerical_indices[:top_n]
        
        poly = PolynomialFeatures(degree=self.config.polynomial_degree, include_bias=False, interaction_only=True)
        X_poly = poly.fit_transform(X[:, selected_indices])
        
        # Combine with original features
        X_combined = np.hstack([X, X_poly[:, len(selected_indices):]])  # Only new features
        
        # Update feature names
        poly_names = poly.get_feature_names_out([feature_names[i] for i in selected_indices])
        new_feature_names = feature_names + list(poly_names[len(selected_indices):])
        
        return X_combined, new_feature_names
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        print("Applying advanced feature engineering...")
        
        # Time features
        df = self.create_time_features(df)
        
        # Interaction features
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = self.create_interaction_features(df, numerical_cols)
        
        # Aggregation features
        df = self.create_aggregation_features(df)
        
        print(f"Feature engineering complete. Shape: {df.shape}")
        return df

# ============================================================================
# ADVANCED LSTM MODELS
# ============================================================================

class AttentionLayer(nn.Module):
    """Attention mechanism for LSTM."""
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights

class BidirectionalLSTMRegressor(nn.Module):
    """Bidirectional LSTM with attention and residual connections."""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout, use_attention=True):
        super(BidirectionalLSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention layer
        if use_attention:
            self.attention = AttentionLayer(hidden_size * 2)  # *2 for bidirectional
        
        # Fully connected layers with residual
        lstm_output_size = hidden_size * 2  # Bidirectional
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x, lengths=None):
        # Pack sequences if lengths provided
        if lengths is not None:
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, c_n) = self.lstm(x_packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use attention or last hidden state
        if self.use_attention:
            context, _ = self.attention(lstm_out)
            x = context
        else:
            # Concatenate forward and backward last hidden states
            x = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        # Fully connected layers with residual
        residual = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load and prepare data."""
    print(f"Loading data from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_excel(filepath)
    df = df.sort_values(by=[config.sequence_id_column, config.date_column])
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def preprocess_data(df: pd.DataFrame, feature_engineer: AdvancedFeatureEngineer) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Preprocess data with feature engineering."""
    print("Preprocessing data...")
    
    # Apply feature engineering
    df = feature_engineer.engineer_features(df)
    
    # Identify columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if config.sequence_id_column in categorical_cols:
        categorical_cols.remove(config.sequence_id_column)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if config.target_column in numerical_cols:
        numerical_cols.remove(config.target_column)
    if config.sequence_id_column in numerical_cols:
        numerical_cols.remove(config.sequence_id_column)
    
    # Remove date columns (already engineered)
    date_cols = [col for col in df.columns if 'Date' in col or col in ['year', 'month', 'day']]
    for col in date_cols:
        if col in numerical_cols:
            numerical_cols.remove(col)
        if col in categorical_cols:
            categorical_cols.remove(col)
    
    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"Numerical columns: {len(numerical_cols)}")
    
    return df, categorical_cols, numerical_cols

def create_preprocessing_pipeline(categorical_cols: List[str], numerical_cols: List[str]) -> ColumnTransformer:
    """Create preprocessing pipeline."""
    numerical_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', RobustScaler())  # More robust to outliers than StandardScaler
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True, max_categories=50))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough',
        n_jobs=-1
    )
    
    return preprocessor

def create_sequences(df: pd.DataFrame, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sequences for LSTM."""
    print("Creating sequences...")
    
    grouped = df.groupby(config.sequence_id_column)
    sequences = []
    targets = []
    sequence_lengths = []
    
    for match_id, group in grouped:
        sequence = group[feature_names].values
        sequences.append(sequence)
        targets.append(group[config.target_column].iloc[0])
        sequence_lengths.append(len(sequence))
    
    max_len = max(sequence_lengths) if sequence_lengths else 1
    X_padded = np.array([
        np.pad(seq, ((0, max_len - len(seq)), (0, 0)), 'constant', constant_values=0) 
        for seq in sequences
    ])
    
    y = np.array(targets, dtype='float32')
    lengths = np.array(sequence_lengths, dtype='int64')
    
    print(f"Created {len(sequences)} sequences, max length: {max_len}")
    return X_padded, y, lengths

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_lstm(model, train_loader, val_loader, criterion, optimizer, epochs, patience, device):
    """Train LSTM model with early stopping."""
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch, lengths in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch, lengths.cpu().numpy())
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch, lengths in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch, lengths.cpu().numpy())
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
    
    return model, train_losses, val_losses

def train_ensemble_models(X_train_flat, y_train, X_val_flat, y_val):
    """Train ensemble of tree-based models."""
    print("Training ensemble models...")
    
    models = {}
    
    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=config.n_estimators_rf,
        max_depth=config.max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=config.random_state,
        n_jobs=-1,
        verbose=0
    )
    rf.fit(X_train_flat, y_train)
    models['rf'] = rf
    
    # Gradient Boosting
    print("Training Gradient Boosting...")
    gb = GradientBoostingRegressor(
        n_estimators=config.n_estimators_gb,
        max_depth=config.max_depth,
        learning_rate=0.1,
        random_state=config.random_state,
        verbose=0
    )
    gb.fit(X_train_flat, y_train)
    models['gb'] = gb
    
    # XGBoost
    print("Training XGBoost...")
    xg = xgb.XGBRegressor(
        n_estimators=config.n_estimators_xgb,
        max_depth=config.max_depth,
        learning_rate=0.1,
        random_state=config.random_state,
        n_jobs=-1,
        verbosity=0
    )
    xg.fit(X_train_flat, y_train)
    models['xgb'] = xg
    
    # Evaluate
    print("\nEnsemble Model Performance:")
    for name, model in models.items():
        y_pred = model.predict(X_val_flat)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        print(f"{name.upper()}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    
    return models

def create_stacked_ensemble(lstm_model, tree_models, X_train_seq, X_train_flat, y_train, 
                           X_val_seq, X_val_flat, y_val, train_lengths, val_lengths, device):
    """Create stacked ensemble combining LSTM and tree models."""
    print("\nCreating stacked ensemble...")
    
    # Get LSTM predictions
    lstm_model.eval()
    with torch.no_grad():
        X_train_seq_tensor = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
        X_val_seq_tensor = torch.tensor(X_val_seq, dtype=torch.float32).to(device)
        
        train_lstm_pred = lstm_model(X_train_seq_tensor, train_lengths).cpu().numpy().flatten()
        val_lstm_pred = lstm_model(X_val_seq_tensor, val_lengths).cpu().numpy().flatten()
    
    # Get tree model predictions
    train_tree_preds = np.column_stack([
        model.predict(X_train_flat) for model in tree_models.values()
    ])
    val_tree_preds = np.column_stack([
        model.predict(X_val_flat) for model in tree_models.values()
    ])
    
    # Combine features for meta-learner
    X_train_meta = np.column_stack([train_lstm_pred, train_tree_preds])
    X_val_meta = np.column_stack([val_lstm_pred, val_tree_preds])
    
    # Train meta-learner (XGBoost)
    meta_learner = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=config.random_state,
        verbosity=0
    )
    meta_learner.fit(X_train_meta, y_train)
    
    # Evaluate
    y_pred_meta = meta_learner.predict(X_val_meta)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_meta))
    mae = mean_absolute_error(y_val, y_pred_meta)
    r2 = r2_score(y_val, y_pred_meta)
    
    print(f"Stacked Ensemble: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    
    return meta_learner

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    start_time = time.time()
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load data
    df = load_data(config.input_file)
    
    # Feature engineering
    feature_engineer = AdvancedFeatureEngineer(config)
    df, categorical_cols, numerical_cols = preprocess_data(df, feature_engineer)
    
    # Preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(categorical_cols, numerical_cols)
    
    # Prepare features
    feature_cols = categorical_cols + numerical_cols
    X = df[feature_cols]
    X_preprocessed = preprocessor.fit_transform(X)
    
    if issparse(X_preprocessed):
        X_preprocessed = X_preprocessed.toarray()
    
    # Get feature names
    cat_encoder = preprocessor.named_transformers_['cat']['onehot']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
    all_feature_names = numerical_cols + list(cat_feature_names)
    
    # Apply polynomial features
    X_preprocessed, all_feature_names = feature_engineer.create_polynomial_features(
        X_preprocessed, all_feature_names
    )
    
    # Create transformed dataframe
    df_transformed = pd.DataFrame(X_preprocessed, columns=all_feature_names, index=df.index)
    df_transformed[config.sequence_id_column] = df[config.sequence_id_column]
    df_transformed[config.target_column] = df[config.target_column]
    
    print(f"Final feature count: {len(all_feature_names)}")
    
    # Create sequences for LSTM
    X_seq, y, sequence_lengths = create_sequences(df_transformed, all_feature_names)
    
    # Split data
    X_train_val_seq, X_test_seq, y_train_val, y_test, train_val_lengths, test_lengths = train_test_split(
        X_seq, y, sequence_lengths, test_size=config.test_size, random_state=config.random_state
    )
    
    X_train_seq, X_val_seq, y_train, y_val, train_lengths, val_lengths = train_test_split(
        X_train_val_seq, y_train_val, train_val_lengths, 
        test_size=config.val_size, random_state=config.random_state
    )
    
    # Flatten sequences for tree models (use last timestep)
    X_train_flat = X_train_seq[:, -1, :]
    X_val_flat = X_val_seq[:, -1, :]
    X_test_flat = X_test_seq[:, -1, :]
    
    print(f"\nData splits:")
    print(f"Train: {len(X_train_seq)}, Val: {len(X_val_seq)}, Test: {len(X_test_seq)}")
    
    # Convert to PyTorch tensors for LSTM
    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(
        X_train_tensor, y_train_tensor, 
        torch.tensor(train_lengths, dtype=torch.int64)
    )
    val_dataset = TensorDataset(
        X_val_tensor, y_val_tensor,
        torch.tensor(val_lengths, dtype=torch.int64)
    )
    test_dataset = TensorDataset(
        X_test_tensor, y_test_tensor,
        torch.tensor(test_lengths, dtype=torch.int64)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.lstm_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.lstm_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.lstm_batch_size, shuffle=False)
    
    # Train LSTM models with different configurations
    print("\n" + "="*80)
    print("Training LSTM Models")
    print("="*80)
    
    best_lstm_model = None
    best_lstm_rmse = float('inf')
    best_lstm_params = None
    
    for hidden_size in config.lstm_hidden_sizes:
        for num_layers in config.lstm_num_layers:
            for dropout in config.lstm_dropout:
                if num_layers == 1 and dropout > 0:
                    continue  # Skip invalid combinations
                
                print(f"\nTraining LSTM: hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout}")
                
                model = BidirectionalLSTMRegressor(
                    input_size=X_train_seq.shape[2],
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    use_attention=True
                ).to(device)
                
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=config.lstm_learning_rate, weight_decay=1e-5)
                
                model, train_losses, val_losses = train_lstm(
                    model, train_loader, val_loader, criterion, optimizer,
                    config.lstm_epochs, config.lstm_patience, device
                )
                
                # Evaluate
                model.eval()
                val_pred = []
                with torch.no_grad():
                    for X_batch, _, lengths in val_loader:
                        outputs = model(X_batch, lengths.cpu().numpy())
                        val_pred.extend(outputs.cpu().numpy().flatten())
                
                val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                print(f"Validation RMSE: {val_rmse:.4f}")
                
                if val_rmse < best_lstm_rmse:
                    best_lstm_rmse = val_rmse
                    best_lstm_model = model
                    best_lstm_params = (hidden_size, num_layers, dropout)
    
    print(f"\nBest LSTM: hidden_size={best_lstm_params[0]}, num_layers={best_lstm_params[1]}, dropout={best_lstm_params[2]}")
    print(f"Best LSTM Validation RMSE: {best_lstm_rmse:.4f}")
    
    # Train ensemble models
    print("\n" + "="*80)
    print("Training Ensemble Models")
    print("="*80)
    
    tree_models = train_ensemble_models(X_train_flat, y_train, X_val_flat, y_val)
    
    # Create stacked ensemble
    print("\n" + "="*80)
    print("Creating Stacked Ensemble")
    print("="*80)
    
    meta_learner = create_stacked_ensemble(
        best_lstm_model, tree_models,
        X_train_seq, X_train_flat, y_train,
        X_val_seq, X_val_flat, y_val,
        train_lengths, val_lengths, device
    )
    
    # Final evaluation on test set
    print("\n" + "="*80)
    print("Final Test Evaluation")
    print("="*80)
    
    # LSTM predictions
    best_lstm_model.eval()
    test_lstm_pred = []
    with torch.no_grad():
        for X_batch, _, lengths in test_loader:
            outputs = best_lstm_model(X_batch, lengths.cpu().numpy())
            test_lstm_pred.extend(outputs.cpu().numpy().flatten())
    test_lstm_pred = np.array(test_lstm_pred)
    
    # Tree model predictions
    test_tree_preds = np.column_stack([
        model.predict(X_test_flat) for model in tree_models.values()
    ])
    
    # Stacked ensemble predictions
    test_meta_features = np.column_stack([test_lstm_pred, test_tree_preds])
    test_meta_pred = meta_learner.predict(test_meta_features)
    
    # Evaluate all models
    print("\nTest Set Results:")
    print(f"LSTM: RMSE={np.sqrt(mean_squared_error(y_test, test_lstm_pred)):.4f}, "
          f"MAE={mean_absolute_error(y_test, test_lstm_pred):.4f}, "
          f"R2={r2_score(y_test, test_lstm_pred):.4f}")
    
    for name, model in tree_models.items():
        pred = model.predict(X_test_flat)
        print(f"{name.upper()}: RMSE={np.sqrt(mean_squared_error(y_test, pred)):.4f}, "
              f"MAE={mean_absolute_error(y_test, pred):.4f}, "
              f"R2={r2_score(y_test, pred):.4f}")
    
    print(f"Stacked Ensemble: RMSE={np.sqrt(mean_squared_error(y_test, test_meta_pred)):.4f}, "
          f"MAE={mean_absolute_error(y_test, test_meta_pred):.4f}, "
          f"R2={r2_score(y_test, test_meta_pred):.4f}")
    
    # Save models
    print("\nSaving models...")
    torch.save(best_lstm_model.state_dict(), 
               os.path.join(config.output_dir, 'best_lstm_model.pth'))
    
    with open(os.path.join(config.output_dir, 'preprocessor.pkl'), 'wb') as f:
        pickle.dump(preprocessor, f)
    
    with open(os.path.join(config.output_dir, 'tree_models.pkl'), 'wb') as f:
        pickle.dump(tree_models, f)
    
    with open(os.path.join(config.output_dir, 'meta_learner.pkl'), 'wb') as f:
        pickle.dump(meta_learner, f)
    
    with open(os.path.join(config.output_dir, 'config.json'), 'w') as f:
        json.dump({
            'best_lstm_params': best_lstm_params,
            'feature_names': all_feature_names[:100],  # Save first 100
            'input_size': X_train_seq.shape[2]
        }, f, indent=2)
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
    print(f"Models saved to: {config.output_dir}")

if __name__ == "__main__":
    main()

