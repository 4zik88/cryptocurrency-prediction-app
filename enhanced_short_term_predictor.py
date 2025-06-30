import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization, 
    MultiHeadAttention, LayerNormalization, Add, 
    Conv1D, GlobalAveragePooling1D, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import RobustScaler
import logging
import os
from typing import Tuple, Optional, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedShortTermPredictor:
    """
    Advanced predictor optimized for short-term (1-hour) cryptocurrency price forecasting.
    
    Features:
    - Multi-head attention mechanisms
    - Multi-timeframe input processing
    - Advanced feature engineering
    - Ensemble prediction methods
    - Robust preprocessing for high-frequency data
    """
    
    def __init__(self, 
                 n_future_steps: int = 1,
                 sequence_length: int = 48,  # Increased for short-term patterns
                 market_type: str = "spot"):
        """
        Initialize the enhanced short-term predictor.
        
        Args:
            n_future_steps: Number of future time steps to predict
            sequence_length: Length of input sequences (hours)
            market_type: "spot" or "futures"
        """
        self.n_future_steps = n_future_steps
        self.sequence_length = sequence_length
        self.market_type = market_type
        self.model = None
        self.ensemble_models = []
        
        # Model architecture parameters
        self.d_model = 128  # Model dimension
        self.num_heads = 8  # Number of attention heads
        self.dropout_rate = 0.1
        self.learning_rate = 0.0001
        
        # Enhanced training parameters for short-term accuracy
        self.batch_size = 64
        self.epochs = 150
        self.patience = 20
        
        # Use RobustScaler for better handling of outliers in high-frequency data
        self.scaler = RobustScaler()
        
        # Model paths
        self.model_path = f"enhanced_short_term_{market_type}_model_{n_future_steps}steps.h5"
        self.weights_path = f"enhanced_short_term_{market_type}_weights_{n_future_steps}steps.h5"
        
        logging.info(f"Initialized EnhancedShortTermPredictor for {market_type} market")
        logging.info(f"Sequence length: {sequence_length}, Future steps: {n_future_steps}")

    def _build_attention_block(self, inputs, name_prefix: str):
        """Build a multi-head attention block with residual connections."""
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            name=f"{name_prefix}_attention"
        )(inputs, inputs)
        
        # Add & Norm
        attention_output = Dropout(self.dropout_rate)(attention_output)
        attention_output = Add()([inputs, attention_output])
        attention_output = LayerNormalization(name=f"{name_prefix}_norm1")(attention_output)
        
        # Feed forward network
        ffn_output = Dense(self.d_model * 2, activation='relu', name=f"{name_prefix}_ffn1")(attention_output)
        ffn_output = Dropout(self.dropout_rate)(ffn_output)
        ffn_output = Dense(self.d_model, name=f"{name_prefix}_ffn2")(ffn_output)
        
        # Add & Norm
        ffn_output = Dropout(self.dropout_rate)(ffn_output)
        ffn_output = Add()([attention_output, ffn_output])
        ffn_output = LayerNormalization(name=f"{name_prefix}_norm2")(ffn_output)
        
        return ffn_output

    def _build_cnn_features(self, inputs, name_prefix: str):
        """Extract local patterns using CNN layers."""
        # Multiple CNN layers for different pattern scales
        conv1 = Conv1D(64, 3, activation='relu', padding='same', name=f"{name_prefix}_conv1")(inputs)
        conv1 = BatchNormalization(name=f"{name_prefix}_bn1")(conv1)
        
        conv2 = Conv1D(64, 5, activation='relu', padding='same', name=f"{name_prefix}_conv2")(inputs)
        conv2 = BatchNormalization(name=f"{name_prefix}_bn2")(conv2)
        
        conv3 = Conv1D(64, 7, activation='relu', padding='same', name=f"{name_prefix}_conv3")(inputs)
        conv3 = BatchNormalization(name=f"{name_prefix}_bn3")(conv3)
        
        # Concatenate different scale features
        cnn_features = Concatenate(name=f"{name_prefix}_concat")([conv1, conv2, conv3])
        return cnn_features

    def build_model(self, n_features: int) -> bool:
        """
        Build the enhanced model architecture with attention and CNN components.
        
        Args:
            n_features: Number of input features
            
        Returns:
            bool: True if model built successfully
        """
        try:
            # Input layer
            inputs = Input(shape=(self.sequence_length, n_features), name="sequence_input")
            
            # Project to model dimension
            projected = Dense(self.d_model, name="input_projection")(inputs)
            
            # Extract CNN features for local patterns
            cnn_features = self._build_cnn_features(projected, "cnn_block")
            
            # Reduce CNN features to model dimension
            cnn_reduced = Dense(self.d_model, name="cnn_reduction")(cnn_features)
            
            # LSTM layers for sequential processing
            lstm1 = LSTM(self.d_model, return_sequences=True, name="lstm1")(projected)
            lstm1 = BatchNormalization(name="lstm1_bn")(lstm1)
            lstm1 = Dropout(self.dropout_rate)(lstm1)
            
            lstm2 = LSTM(self.d_model, return_sequences=True, name="lstm2")(lstm1)
            lstm2 = BatchNormalization(name="lstm2_bn")(lstm2)
            lstm2 = Dropout(self.dropout_rate)(lstm2)
            
            # Combine LSTM and CNN features
            combined_features = Add(name="combine_features")([lstm2, cnn_reduced])
            
            # Multi-head attention blocks
            attention1 = self._build_attention_block(combined_features, "attention1")
            attention2 = self._build_attention_block(attention1, "attention2")
            
            # Global pooling and dense layers
            pooled = GlobalAveragePooling1D(name="global_pooling")(attention2)
            
            # Dense layers for final prediction
            dense1 = Dense(256, activation='relu', name="dense1")(pooled)
            dense1 = BatchNormalization(name="dense1_bn")(dense1)
            dense1 = Dropout(self.dropout_rate)(dense1)
            
            dense2 = Dense(128, activation='relu', name="dense2")(dense1)
            dense2 = BatchNormalization(name="dense2_bn")(dense2)
            dense2 = Dropout(self.dropout_rate)(dense2)
            
            dense3 = Dense(64, activation='relu', name="dense3")(dense2)
            dense3 = Dropout(self.dropout_rate)(dense3)
            
            # Output layer
            outputs = Dense(self.n_future_steps, activation='linear', name="output")(dense3)
            
            # Create model
            self.model = Model(inputs=inputs, outputs=outputs, name="EnhancedShortTermPredictor")
            
            # Advanced optimizer configuration
            optimizer = Adam(
                learning_rate=self.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                clipnorm=1.0  # Gradient clipping for stability
            )
            
            # Compile with multiple metrics
            self.model.compile(
                optimizer=optimizer,
                loss='huber',  # More robust to outliers than MSE
                metrics=['mae', 'mse']
            )
            
            logging.info("Successfully built enhanced short-term prediction model")
            logging.info(f"Model parameters: {self.model.count_params():,}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error building enhanced model: {str(e)}")
            return False

    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> bool:
        """
        Train the enhanced model with advanced callbacks and techniques.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            
        Returns:
            bool: True if training successful
        """
        try:
            if self.model is None:
                n_features = X_train.shape[2]
                if not self.build_model(n_features):
                    return False
            
            # Prepare validation data
            if X_val is None or y_val is None:
                val_split = 0.15  # Smaller validation split for more training data
                val_size = int(len(X_train) * val_split)
                X_val = X_train[-val_size:]
                y_val = y_train[-val_size:]
                X_train = X_train[:-val_size]
                y_train = y_train[:-val_size]
            
            # Advanced callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.patience,
                    restore_best_weights=True,
                    verbose=1,
                    mode='min'
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.patience // 2,
                    min_lr=1e-7,
                    verbose=1,
                    mode='min'
                ),
                ModelCheckpoint(
                    self.weights_path,
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1,
                    mode='min'
                )
            ]
            
            logging.info(f"Starting enhanced model training...")
            logging.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
            
            # Train the model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )
            
            # Save the full model
            self.save_model()
            
            # Log training results
            final_loss = min(history.history['val_loss'])
            final_mae = min(history.history['val_mae'])
            logging.info(f"Training completed. Best val_loss: {final_loss:.6f}, Best val_mae: {final_mae:.6f}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error training enhanced model: {str(e)}")
            return False

    def predict(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Make predictions using the trained model."""
        try:
            if self.model is None:
                logging.error("Model not loaded. Please train or load a model first.")
                return None
            
            predictions = self.model.predict(X, batch_size=self.batch_size, verbose=0)
            logging.info(f"Generated {len(predictions)} enhanced predictions")
            return predictions
            
        except Exception as e:
            logging.error(f"Error making enhanced predictions: {str(e)}")
            return None

    def predict_with_uncertainty(self, 
                                X: np.ndarray, 
                                n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimation using Monte Carlo dropout.
        
        Args:
            X: Input sequences
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Tuple of (mean_predictions, lower_bound, upper_bound)
        """
        try:
            if self.model is None:
                logging.error("Model not loaded. Please train or load a model first.")
                return None, None, None
            
            # Enable dropout during inference
            predictions = []
            for _ in range(n_samples):
                pred = self.model(X, training=True)
                predictions.append(pred.numpy())
            
            predictions = np.array(predictions)
            
            # Calculate statistics
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            
            # 95% confidence intervals
            lower_bound = mean_pred - 1.96 * std_pred
            upper_bound = mean_pred + 1.96 * std_pred
            
            logging.info("Generated predictions with uncertainty bounds")
            return mean_pred, lower_bound, upper_bound
            
        except Exception as e:
            logging.error(f"Error making predictions with uncertainty: {str(e)}")
            return None, None, None

    def save_model(self):
        """Save the trained model."""
        if self.model is None:
            logging.warning("Attempted to save a model that is None.")
            return False
        try:
            self.model.save(self.model_path)
            logging.info(f"Enhanced model saved to {self.model_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving enhanced model: {str(e)}")
            return False

    def load_model(self, n_features: int) -> bool:
        """Load a pre-trained model if compatible."""
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                
                # Check compatibility
                expected_shape = (None, self.sequence_length, n_features)
                actual_shape = self.model.input_shape
                
                if actual_shape == expected_shape:
                    logging.info(f"Compatible enhanced model loaded from {self.model_path}")
                    return True
                else:
                    logging.warning(f"Model shape mismatch. Expected {expected_shape}, got {actual_shape}")
                    return False
                    
            except Exception as e:
                logging.error(f"Error loading enhanced model: {str(e)}")
                return False
        
        # Try to load weights if full model not available
        if os.path.exists(self.weights_path):
            try:
                if self.build_model(n_features):
                    self.model.load_weights(self.weights_path)
                    logging.info(f"Enhanced model weights loaded from {self.weights_path}")
                    return True
            except Exception as e:
                logging.error(f"Error loading enhanced model weights: {str(e)}")
        
        logging.info("No compatible enhanced model found, will train new model")
        return False

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Optional[Dict]:
        """Comprehensive model evaluation with multiple metrics."""
        try:
            if self.model is None:
                logging.error("Model not loaded. Please train or load a model first.")
                return None
            
            # Make predictions
            predictions = self.predict(X_test)
            if predictions is None:
                return None
            
            # Calculate comprehensive metrics
            mse = np.mean((predictions - y_test) ** 2)
            mae = np.mean(np.abs(predictions - y_test))
            rmse = np.sqrt(mse)
            
            # Percentage errors
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            
            # Directional accuracy (crucial for trading)
            if len(y_test) > 1:
                actual_direction = np.sign(np.diff(y_test.flatten()))
                predicted_direction = np.sign(np.diff(predictions.flatten()))
                directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
            else:
                directional_accuracy = 0
            
            # R-squared
            ss_res = np.sum((y_test - predictions) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'directional_accuracy': float(directional_accuracy),
                'r2_score': float(r2_score)
            }
            
            logging.info("Enhanced Model Evaluation:")
            for metric, value in metrics.items():
                logging.info(f"  {metric.upper()}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error evaluating enhanced model: {str(e)}")
            return None 