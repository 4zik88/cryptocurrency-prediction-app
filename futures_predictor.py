import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class FuturesLSTMPredictor:
    def __init__(self, n_future_steps=1, sequence_length=24):
        """
        Initialize the Futures LSTM Predictor with enhanced architecture.
        
        Args:
            n_future_steps: Number of future time steps to predict
            sequence_length: Length of input sequences
        """
        self.n_future_steps = n_future_steps
        self.sequence_length = sequence_length
        self.model = None
        self.model_path = f"futures_lstm_model_{n_future_steps}steps.h5"
        
        # Enhanced model parameters for futures
        self.lstm_units = [64, 32]  # Multiple LSTM layers
        self.dropout_rate = 0.3
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 100
        
        logging.info(f"Initialized FuturesLSTMPredictor for {n_future_steps} future steps")

    def build_model(self, n_features):
        """Build enhanced LSTM model architecture for futures prediction."""
        try:
            model = Sequential()
            
            # First LSTM layer with return sequences
            model.add(LSTM(
                units=self.lstm_units[0],
                return_sequences=True,
                input_shape=(self.sequence_length, n_features),
                name='lstm_1'
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
            
            # Second LSTM layer
            model.add(LSTM(
                units=self.lstm_units[1],
                return_sequences=False,
                name='lstm_2'
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
            
            # Dense layers for futures-specific processing
            model.add(Dense(32, activation='relu', name='dense_1'))
            model.add(Dropout(self.dropout_rate / 2))
            
            model.add(Dense(16, activation='relu', name='dense_2'))
            model.add(Dropout(self.dropout_rate / 2))
            
            # Output layer
            model.add(Dense(self.n_future_steps, activation='linear', name='output'))
            
            # Compile with advanced optimizer settings
            optimizer = Adam(
                learning_rate=self.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            )
            
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae', 'mape']
            )
            
            self.model = model
            logging.info(f"Successfully built futures LSTM model for {self.n_future_steps} future steps.")
            logging.info(f"Model architecture: {len(self.lstm_units)} LSTM layers, {n_features} features")
            
            return True
            
        except Exception as e:
            logging.error(f"Error building futures model: {str(e)}")
            return False

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the futures LSTM model with advanced callbacks."""
        try:
            if self.model is None:
                n_features = X_train.shape[2]
                if not self.build_model(n_features):
                    return False
            
            # Prepare validation data
            if X_val is None or y_val is None:
                # Use 20% of training data for validation
                val_split = 0.2
                val_size = int(len(X_train) * val_split)
                X_val = X_train[-val_size:]
                y_val = y_train[-val_size:]
                X_train = X_train[:-val_size]
                y_train = y_train[:-val_size]
            
            # Advanced callbacks for futures training
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
            
            logging.info(f"Starting futures model training...")
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
            
            # Save the trained model
            self.save_model()
            
            # Log training results
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            logging.info(f"Training completed. Final loss: {final_loss:.6f}, Final val_loss: {final_val_loss:.6f}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error training futures model: {str(e)}")
            return False

    def predict(self, X):
        """Make predictions using the trained futures model."""
        try:
            if self.model is None:
                logging.error("Model not loaded. Please train or load a model first.")
                return None
            
            predictions = self.model.predict(X, verbose=0)
            logging.info(f"Generated {len(predictions)} futures predictions")
            return predictions
            
        except Exception as e:
            logging.error(f"Error making futures predictions: {str(e)}")
            return None

    def save_model(self):
        """Save the trained futures model."""
        try:
            if self.model is not None:
                self.model.save(self.model_path)
                logging.info(f"Futures model saved to {self.model_path}")
                return True
            else:
                logging.warning("No model to save")
                return False
        except Exception as e:
            logging.error(f"Error saving futures model: {str(e)}")
            return False

    def load_model(self):
        """Load a pre-trained futures model."""
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                logging.info(f"Futures model loaded from {self.model_path}")
                return True
            else:
                logging.info(f"No saved futures model found at {self.model_path}")
                return False
        except Exception as e:
            logging.error(f"Error loading futures model: {str(e)}")
            return False

    def evaluate_model(self, X_test, y_test):
        """Evaluate the futures model performance."""
        try:
            if self.model is None:
                logging.error("Model not loaded. Please train or load a model first.")
                return None
            
            # Make predictions
            predictions = self.predict(X_test)
            if predictions is None:
                return None
            
            # Calculate metrics
            mse = np.mean((predictions - y_test) ** 2)
            mae = np.mean(np.abs(predictions - y_test))
            rmse = np.sqrt(mse)
            
            # Calculate percentage errors
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            
            # Calculate directional accuracy (important for futures trading)
            actual_direction = np.sign(np.diff(y_test.flatten()))
            predicted_direction = np.sign(np.diff(predictions.flatten()))
            directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'directional_accuracy': directional_accuracy
            }
            
            logging.info(f"Futures Model Evaluation:")
            logging.info(f"  MSE: {mse:.6f}")
            logging.info(f"  MAE: {mae:.6f}")
            logging.info(f"  RMSE: {rmse:.6f}")
            logging.info(f"  MAPE: {mape:.2f}%")
            logging.info(f"  Directional Accuracy: {directional_accuracy:.2f}%")
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error evaluating futures model: {str(e)}")
            return None

    def get_model_summary(self):
        """Get a summary of the futures model architecture."""
        try:
            if self.model is not None:
                return self.model.summary()
            else:
                logging.warning("No model available for summary")
                return None
        except Exception as e:
            logging.error(f"Error getting futures model summary: {str(e)}")
            return None

    def predict_with_confidence(self, X, n_simulations=100):
        """
        Make predictions with confidence intervals using Monte Carlo dropout.
        Useful for futures trading risk assessment.
        """
        try:
            if self.model is None:
                logging.error("Model not loaded. Please train or load a model first.")
                return None, None, None
            
            # Enable dropout during inference for uncertainty estimation
            predictions = []
            for _ in range(n_simulations):
                # Make prediction with dropout enabled
                pred = self.model(X, training=True)
                predictions.append(pred.numpy())
            
            predictions = np.array(predictions)
            
            # Calculate statistics
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            
            # Calculate confidence intervals (95%)
            confidence_lower = mean_pred - 1.96 * std_pred
            confidence_upper = mean_pred + 1.96 * std_pred
            
            logging.info(f"Generated futures predictions with confidence intervals")
            
            return mean_pred, confidence_lower, confidence_upper
            
        except Exception as e:
            logging.error(f"Error making futures predictions with confidence: {str(e)}")
            return None, None, None 