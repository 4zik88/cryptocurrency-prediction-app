import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class LSTMPredictor:
    def __init__(self, sequence_length=24, n_future_steps=1):
        self.sequence_length = sequence_length
        self.n_future_steps = n_future_steps
        self.model_path = f'crypto_lstm_model_seq{sequence_length}_future{n_future_steps}.h5'
        self.model = self._build_model()
        
    def _build_model(self):
        """Build LSTM model architecture for multi-step forecasting."""
        model = Sequential([
            LSTM(units=100, return_sequences=True, input_shape=(self.sequence_length, 6)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=32, activation='relu'),
            Dense(units=self.n_future_steps)
        ])
        
        # Use standard Adam optimizer (Keras 3 compatible)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        logging.info(f"Successfully built LSTM model for {self.n_future_steps} future steps.")
        return model
    
    def train(self, X_train, y_train, epochs=20, batch_size=32, validation_split=0.1):
        """Train the LSTM model."""
        logging.info(f"Starting model training for {self.n_future_steps}-step forecast...")
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
        )
        
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        logging.info(f"Training completed - Loss: {final_loss:.4f}, Val Loss: {final_val_loss:.4f}")
        self.save_model()
    
    def predict(self, X):
        """Generate predictions for input sequences."""
        logging.info(f"Generating predictions for {len(X)} sequences.")
        return self.model.predict(X, verbose=0)
    
    def save_model(self):
        """Save the model to a file."""
        try:
            self.model.save(self.model_path)
            logging.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
    
    def load_model(self):
        """Load a saved model."""
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                logging.info(f"Model loaded from {self.model_path}")
                return True
            except Exception as e:
                logging.error(f"Error loading model: {str(e)}")
                return False
        return False

    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        if self.model is None:
            logging.error("Model not initialized")
            return None
            
        try:
            loss = self.model.evaluate(X_test, y_test, verbose=0)
            logging.info(f"Model evaluation - Test Loss: {loss:.4f}")
            return loss
            
        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            return None
    
    def predict_next(self, last_sequence):
        """Predict the next value given the last sequence."""
        if self.model is None:
            logging.error("Model not initialized")
            return None
            
        try:
            prediction = self.model.predict(last_sequence, verbose=0)
            logging.info("Generated prediction for next timestep")
            return prediction[0][0]
            
        except Exception as e:
            logging.error(f"Error predicting next value: {str(e)}")
            return None 