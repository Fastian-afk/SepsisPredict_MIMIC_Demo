import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
import logging
import joblib
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

class MIMICSepsisPredictor:
    def __init__(self, data_path=None, model_type='random_forest', model_params=None):
        """
        Initialize the MIMIC sepsis predictor
        
        Args:
            data_path (str): Path to the MIMIC data files
            model_type (str): Type of model to train ('random_forest', 'gradient_boosting', 'logistic_regression', 'svm')
            model_params (dict): Dictionary of model parameters
        """
        self.data_path = data_path
        self.model_type = model_type
        self.model_params = model_params or {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.target_column = None
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the model based on the specified type"""
        models = {
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'logistic_regression': LogisticRegression,
            'svm': SVC
        }
        
        if self.model_type not in models:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        return models[self.model_type](**self.model_params)
        
    def load_data(self, file_path):
        """Load data from a CSV file"""
        try:
            logger.info(f"Loading data from {file_path}")
            data = pd.read_csv(file_path)
            logger.info(f"Successfully loaded data with shape {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def _extract_lab_values(self, medical_report):
        """Extract laboratory values from medical report text"""
        lab_values = {}
        
        # Extract lab results section
        lab_section = re.search(r'\*\*Lab Results:\*\*(.*?)(?=\n\n|$)', medical_report, re.DOTALL)
        if lab_section:
            lab_text = lab_section.group(1)
            
            # Extract each lab value
            lab_pattern = r'- ([^:]+): ([0-9.]+) ([^()]+)'
            matches = re.finditer(lab_pattern, lab_text)
            
            for match in matches:
                test_name = match.group(1).strip()
                value = float(match.group(2))
                unit = match.group(3).strip()
                lab_values[test_name] = value
                
        return lab_values
        
    def _extract_patient_info(self, medical_report):
        """Extract patient information from medical report text"""
        info = {}
        
        # Extract age
        age_match = re.search(r'Age: (\d+)', medical_report)
        if age_match:
            info['age'] = int(age_match.group(1))
            
        # Extract gender
        gender_match = re.search(r'Gender: ([MF])', medical_report)
        if gender_match:
            info['gender'] = gender_match.group(1)
            
        # Extract admission type
        adm_type_match = re.search(r'Admission Type: ([A-Z]+)', medical_report)
        if adm_type_match:
            info['admission_type'] = adm_type_match.group(1)
            
        return info
        
    def _create_sepsis_target(self, lab_values):
        """
        Create sepsis target based on lab values
        Criteria for sepsis (very lenient):
        - WBC > 9 or < 5
        - Heart rate > 80
        - Respiratory rate > 16
        - Temperature > 37.2 or < 36.8
        - Lactate > 1.5
        """
        sepsis_indicators = 0
        indicators_present = []
        
        # Print available lab values for debugging
        logger.debug(f"Available lab values: {lab_values}")
        
        # Check WBC
        if 'White Blood Cells' in lab_values:
            wbc = lab_values['White Blood Cells']
            if wbc > 9 or wbc < 5:
                sepsis_indicators += 1
                indicators_present.append(f'WBC: {wbc}')
                
        # Check Heart Rate
        if 'Heart Rate' in lab_values:
            hr = lab_values['Heart Rate']
            if hr > 80:
                sepsis_indicators += 1
                indicators_present.append(f'HR: {hr}')
                
        # Check Respiratory Rate
        if 'Respiratory Rate' in lab_values:
            rr = lab_values['Respiratory Rate']
            if rr > 16:
                sepsis_indicators += 1
                indicators_present.append(f'RR: {rr}')
                
        # Check Temperature
        if 'Temperature' in lab_values:
            temp = lab_values['Temperature']
            if temp > 37.2 or temp < 36.8:
                sepsis_indicators += 1
                indicators_present.append(f'Temp: {temp}')
                
        # Check Lactate
        if 'Lactate' in lab_values:
            lactate = lab_values['Lactate']
            if lactate > 1.5:
                sepsis_indicators += 1
                indicators_present.append(f'Lactate: {lactate}')
                
        # If 1 or more indicators are present, consider it sepsis (very lenient)
        is_sepsis = 1 if sepsis_indicators >= 1 else 0
        
        # Log the indicators for debugging
        if is_sepsis:
            logger.info(f"Sepsis indicators present: {', '.join(indicators_present)}")
            
        return is_sepsis
        
    def preprocess_data(self, data):
        """Preprocess the data"""
        try:
            logger.info("Starting data preprocessing")
            
            # Extract features from medical reports
            processed_data = []
            sepsis_count = 0
            total_count = 0
            
            # Print first few medical reports for debugging
            logger.info("\nSample medical reports:")
            for i, (_, row) in enumerate(data.iterrows()):
                if i < 3:  # Print first 3 reports
                    logger.info(f"\nReport {i+1}:")
                    logger.info(row['medical_report'])
                    
            for _, row in data.iterrows():
                lab_values = self._extract_lab_values(row['medical_report'])
                patient_info = self._extract_patient_info(row['medical_report'])
                
                # Create sepsis target
                sepsis = self._create_sepsis_target(lab_values)
                if sepsis == 1:
                    sepsis_count += 1
                total_count += 1
                
                # Combine all features
                features = {**lab_values, **patient_info, 'sepsis': sepsis}
                processed_data.append(features)
                
            # Log sepsis statistics
            logger.info(f"\nTotal cases: {total_count}")
            logger.info(f"Sepsis cases: {sepsis_count}")
            logger.info(f"Sepsis percentage: {(sepsis_count/total_count)*100:.2f}%")
                
            # Convert to DataFrame
            processed_df = pd.DataFrame(processed_data)
            
            # Handle missing values
            processed_df = self._handle_missing_values(processed_df)
            
            # Encode categorical variables
            processed_df = self._encode_categorical_variables(processed_df)
            
            # Scale numerical features
            processed_df = self._scale_numerical_features(processed_df)
            
            # Set feature columns
            self.feature_columns = [col for col in processed_df.columns if col != 'sepsis']
            
            # Split features and target
            X = processed_df[self.feature_columns]
            y = processed_df['sepsis']
            
            logger.info("Data preprocessing completed successfully")
            return X, y
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
            
    def _handle_missing_values(self, data):
        """Handle missing values in the dataset"""
        # Fill numerical columns with median
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())
        
        # Fill categorical columns with mode
        categorical_cols = data.select_dtypes(include=['object']).columns
        data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
        
        return data
        
    def _encode_categorical_variables(self, data):
        """Encode categorical variables"""
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != 'sepsis':
                data[col] = self.label_encoder.fit_transform(data[col])
                
        return data
        
    def _scale_numerical_features(self, data):
        """Scale numerical features"""
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        if 'sepsis' in numerical_cols:
            numerical_cols = numerical_cols.drop('sepsis')
            
        data[numerical_cols] = self.scaler.fit_transform(data[numerical_cols])
        
        return data
        
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        try:
            logger.info("Splitting data into training and testing sets")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Apply SMOTE to balance the training data
            smote = SMOTE(random_state=random_state)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            
            logger.info(f"Training set shape: {X_train_balanced.shape}, Testing set shape: {X_test.shape}")
            return X_train_balanced, X_test, y_train_balanced, y_test
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
            
    def train(self, X_train, y_train):
        """Train the model"""
        try:
            logger.info(f"Training {self.model_type} model")
            self.model.fit(X_train, y_train)
            logger.info("Model training completed successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
            
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data"""
        try:
            logger.info("Evaluating model performance")
            y_pred = self.model.predict(X_test)
            
            # Get prediction probabilities if available
            if hasattr(self.model, 'predict_proba'):
                y_pred_proba = self.model.predict_proba(X_test)
                if y_pred_proba.shape[1] > 1:  # Check if we have probabilities for both classes
                    y_pred_proba = y_pred_proba[:, 1]
                else:
                    y_pred_proba = y_pred_proba[:, 0]
            else:
                y_pred_proba = None
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
            
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                
            # Add confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics['confusion_matrix'] = cm
            
            # Create visualizations
            self._create_evaluation_plots(y_test, y_pred, y_pred_proba, cm)
                
            logger.info("Model evaluation completed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
            
    def _create_evaluation_plots(self, y_test, y_pred, y_pred_proba, cm):
        """Create evaluation plots"""
        try:
            # 1. Confusion Matrix Heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('plots/confusion_matrix.png')
            plt.close()
            
            # 2. ROC Curve
            if y_pred_proba is not None:
                plt.figure(figsize=(8, 6))
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig('plots/roc_curve.png')
                plt.close()
            
            # 3. Feature Importance (if using Random Forest)
            if hasattr(self.model, 'feature_importances_'):
                plt.figure(figsize=(12, 6))
                importances = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': self.model.feature_importances_
                })
                importances = importances.sort_values('importance', ascending=False).head(10)
                plt.barh(importances['feature'], importances['importance'])
                plt.title('Top 10 Most Important Features')
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig('plots/feature_importance.png')
                plt.close()
            
            logger.info("Evaluation plots created successfully")
            
        except Exception as e:
            logger.error(f"Error creating evaluation plots: {str(e)}")
            raise
            
    def predict(self, X):
        """Make predictions on new data"""
        try:
            logger.info("Making predictions")
            predictions = self.model.predict(X)
            prediction_probas = self.model.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else None
            
            return predictions, prediction_probas
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
            
    def save_model(self, model_path):
        """Save the trained model to disk"""
        try:
            logger.info(f"Saving model to {model_path}")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(self.model, model_path)
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self, model_path):
        """Load a trained model from disk"""
        try:
            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            logger.info("Model loaded successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

def main():
    # Example usage
    predictor = MIMICSepsisPredictor(
        model_type='random_forest',
        model_params={'n_estimators': 100, 'random_state': 42, 'class_weight': 'balanced'}
    )
    
    # Load and preprocess data
    data = predictor.load_data('Data/structured_medical_records.csv')
    X, y = predictor.preprocess_data(data)
    
    # Print class distribution
    print("\nClass Distribution:")
    print(y.value_counts(normalize=True))
    
    # Split data
    X_train, X_test, y_train, y_test = predictor.split_data(X, y)
    
    # Train model
    predictor.train(X_train, y_train)
    
    # Evaluate model
    metrics = predictor.evaluate(X_test, y_test)
    print("\nModel Performance Metrics:")
    for metric_name, value in metrics.items():
        if metric_name != 'confusion_matrix':
            print(f"{metric_name}: {value:.4f}")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Save model
    predictor.save_model('models/sepsis_predictor.joblib')
    
    print("\nPlots have been saved in the 'plots' directory:")
    print("- confusion_matrix.png")
    print("- roc_curve.png")
    print("- feature_importance.png")

if __name__ == "__main__":
    main() 