import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import logging
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedCryptoPredictor:
    def __init__(self, sequence_length=24, n_future_steps=1):
        """
        Enhanced crypto predictor with ensemble methods and uncertainty quantification.
        
        Args:
            sequence_length: Length of input sequences
            n_future_steps: Number of future time steps to predict
        """
        self.sequence_length = sequence_length
        self.n_future_steps = n_future_steps
        self.lstm_model = None
        self.rf_model = None
        self.ensemble_weights = [0.7, 0.3]  # LSTM, Random Forest
        
        # Model paths
        self.lstm_model_path = f'enhanced_lstm_model_seq{sequence_length}_future{n_future_steps}.h5'
        self.rf_model_path = f'enhanced_rf_model_seq{sequence_length}_future{n_future_steps}.pkl'
        
        # Training parameters
        self.lstm_units = [128, 64, 32]
        self.dropout_rate = 0.3
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 100
        
        logging.info(f"Initialized EnhancedCryptoPredictor for {n_future_steps} future steps")

    def _build_lstm_model(self, n_features):
        """Build enhanced LSTM model with uncertainty quantification."""
        try:
            model = Sequential([
                LSTM(units=self.lstm_units[0], return_sequences=True, 
                     input_shape=(self.sequence_length, n_features)),
                BatchNormalization(),
                Dropout(self.dropout_rate),
                
                LSTM(units=self.lstm_units[1], return_sequences=True),
                BatchNormalization(),
                Dropout(self.dropout_rate),
                
                LSTM(units=self.lstm_units[2], return_sequences=False),
                BatchNormalization(),
                Dropout(self.dropout_rate),
                
                Dense(64, activation='relu'),
                Dropout(self.dropout_rate / 2),
                
                Dense(32, activation='relu'),
                Dropout(self.dropout_rate / 2),
                
                Dense(self.n_future_steps, activation='linear')
            ])
            
            # Compile model
            optimizer = Adam(learning_rate=self.learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']
            )
            
            self.lstm_model = model
            logging.info(f"Successfully built enhanced LSTM model with {n_features} features")
            return True
            
        except Exception as e:
            logging.error(f"Error building LSTM model: {str(e)}")
            return False

    def _build_rf_model(self):
        """Build Random Forest model for ensemble."""
        try:
            self.rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            logging.info("Successfully built Random Forest model")
            return True
        except Exception as e:
            logging.error(f"Error building RF model: {str(e)}")
            return False

    def _prepare_rf_data(self, X):
        """Prepare data for Random Forest (flatten sequences)."""
        try:
            # Flatten the sequences for RF
            n_samples, seq_len, n_features = X.shape
            X_flat = X.reshape(n_samples, seq_len * n_features)
            return X_flat
        except Exception as e:
            logging.error(f"Error preparing RF data: {str(e)}")
            return None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the ensemble model."""
        try:
            n_features = X_train.shape[2]
            
            # Prepare validation data
            if X_val is None or y_val is None:
                val_split = 0.2
                val_size = int(len(X_train) * val_split)
                X_val = X_train[-val_size:]
                y_val = y_train[-val_size:]
                X_train = X_train[:-val_size]
                y_train = y_train[:-val_size]
            
            # Build models
            if not self._build_lstm_model(n_features):
                return False
            if not self._build_rf_model():
                return False
            
            # Train LSTM model
            logging.info("Training LSTM model...")
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=8,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            
            history = self.lstm_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )
            
            # Train Random Forest model
            logging.info("Training Random Forest model...")
            X_train_flat = self._prepare_rf_data(X_train)
            X_val_flat = self._prepare_rf_data(X_val)
            
            if X_train_flat is not None:
                self.rf_model.fit(X_train_flat, y_train)
                
                # Evaluate RF model
                val_pred_rf = self.rf_model.predict(X_val_flat)
                rf_mse = mean_squared_error(y_val, val_pred_rf)
                rf_mae = mean_absolute_error(y_val, val_pred_rf)
                logging.info(f"Random Forest - MSE: {rf_mse:.6f}, MAE: {rf_mae:.6f}")
            
            # Save models
            self.save_model()
            
            # Log training results
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            logging.info(f"LSTM Training completed. Final loss: {final_loss:.6f}, Final val_loss: {final_val_loss:.6f}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error training ensemble model: {str(e)}")
            return False

    def predict_with_uncertainty(self, X, n_simulations=100):
        """Generate predictions with uncertainty quantification."""
        try:
            if self.lstm_model is None or self.rf_model is None:
                logging.error("Models not loaded. Please train or load models first.")
                return None
            
            # LSTM predictions with uncertainty (Monte Carlo Dropout)
            lstm_predictions = []
            for _ in range(n_simulations):
                # Use dropout during inference for uncertainty estimation
                pred = self.lstm_model(X, training=True)
                lstm_predictions.append(pred.numpy())
            
            lstm_predictions = np.array(lstm_predictions)
            
            # Random Forest predictions
            X_flat = self._prepare_rf_data(X)
            rf_pred = self.rf_model.predict(X_flat)
            
            # Ensemble predictions
            lstm_mean = np.mean(lstm_predictions, axis=0)
            ensemble_pred = (self.ensemble_weights[0] * lstm_mean + 
                           self.ensemble_weights[1] * rf_pred)
            
            # Calculate uncertainty metrics
            uncertainty_metrics = {
                'mean_prediction': ensemble_pred,
                'lstm_mean': lstm_mean,
                'lstm_std': np.std(lstm_predictions, axis=0),
                'lstm_predictions': lstm_predictions,
                'rf_prediction': rf_pred,
                'confidence_95_lower': np.percentile(lstm_predictions, 2.5, axis=0),
                'confidence_95_upper': np.percentile(lstm_predictions, 97.5, axis=0),
                'confidence_68_lower': np.percentile(lstm_predictions, 16, axis=0),
                'confidence_68_upper': np.percentile(lstm_predictions, 84, axis=0),
                'prediction_variance': np.var(lstm_predictions, axis=0),
                'model_disagreement': np.abs(lstm_mean - rf_pred)
            }
            
            logging.info(f"Generated predictions with uncertainty for {len(X)} samples")
            return uncertainty_metrics
            
        except Exception as e:
            logging.error(f"Error making predictions with uncertainty: {str(e)}")
            return None

    def predict(self, X):
        """Generate point predictions (backward compatibility)."""
        try:
            uncertainty_result = self.predict_with_uncertainty(X, n_simulations=10)
            if uncertainty_result is not None:
                return uncertainty_result['mean_prediction']
            return None
        except Exception as e:
            logging.error(f"Error making predictions: {str(e)}")
            return None

    def save_model(self):
        """Save both models."""
        try:
            # Save LSTM model
            if self.lstm_model is not None:
                self.lstm_model.save(self.lstm_model_path)
                logging.info(f"LSTM model saved to {self.lstm_model_path}")
            
            # Save Random Forest model
            if self.rf_model is not None:
                with open(self.rf_model_path, 'wb') as f:
                    pickle.dump(self.rf_model, f)
                logging.info(f"Random Forest model saved to {self.rf_model_path}")
            
            return True
        except Exception as e:
            logging.error(f"Error saving models: {str(e)}")
            return False

    def load_model(self):
        """Load both models."""
        try:
            success = True
            
            # Load LSTM model
            if os.path.exists(self.lstm_model_path):
                self.lstm_model = tf.keras.models.load_model(self.lstm_model_path)
                logging.info(f"LSTM model loaded from {self.lstm_model_path}")
            else:
                logging.info(f"No LSTM model found at {self.lstm_model_path}")
                success = False
            
            # Load Random Forest model
            if os.path.exists(self.rf_model_path):
                with open(self.rf_model_path, 'rb') as f:
                    self.rf_model = pickle.load(f)
                logging.info(f"Random Forest model loaded from {self.rf_model_path}")
            else:
                logging.info(f"No Random Forest model found at {self.rf_model_path}")
                success = False
            
            return success
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            return False

    def evaluate_ensemble(self, X_test, y_test):
        """Evaluate ensemble model performance."""
        try:
            if self.lstm_model is None or self.rf_model is None:
                logging.error("Models not loaded")
                return None
            
            # Get predictions
            uncertainty_result = self.predict_with_uncertainty(X_test, n_simulations=50)
            if uncertainty_result is None:
                return None
            
            predictions = uncertainty_result['mean_prediction']
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mse)
            
            # Calculate prediction intervals coverage
            lower_95 = uncertainty_result['confidence_95_lower']
            upper_95 = uncertainty_result['confidence_95_upper']
            coverage_95 = np.mean((y_test >= lower_95) & (y_test <= upper_95))
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'coverage_95': coverage_95,
                'mean_prediction_std': np.mean(uncertainty_result['lstm_std']),
                'mean_model_disagreement': np.mean(uncertainty_result['model_disagreement'])
            }
            
            logging.info(f"Ensemble evaluation - MSE: {mse:.6f}, MAE: {mae:.6f}, Coverage 95%: {coverage_95:.3f}")
            return metrics
            
        except Exception as e:
            logging.error(f"Error evaluating ensemble: {str(e)}")
            return None

    def get_feature_importance(self):
        """Get feature importance from Random Forest model."""
        try:
            if self.rf_model is None:
                logging.error("Random Forest model not loaded")
                return None
            
            importance = self.rf_model.feature_importances_
            logging.info("Successfully retrieved feature importance")
            return importance
        except Exception as e:
            logging.error(f"Error getting feature importance: {str(e)}")
            return None 