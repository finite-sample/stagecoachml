"""Handwritten digits recognition pipeline using StagecoachML.

This example demonstrates a computer vision workflow:
1. Load digits dataset from sklearn (8x8 pixel images)
2. Visualize sample digits
3. Preprocess and split data
4. Train multiple classification models
5. Evaluate with confusion matrices
6. Test on individual digit predictions

Run with: python examples/digits_recognition/digits_pipeline.py
"""

import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from stagecoachml import Pipeline
from stagecoachml.stage import FunctionStage


def main():
    """Run the complete digits recognition pipeline."""
    print("🔢 Handwritten Digits Recognition with StagecoachML")
    print("=" * 55)

    # Create pipeline
    pipeline = Pipeline(name="digits_recognition")

    # Stage 1: Load digits dataset
    def load_digits_data(context):
        """Load the handwritten digits dataset from sklearn."""
        digits = load_digits()
        
        print(f"📊 Loaded digits dataset:")
        print(f"   Images: {digits.images.shape[0]} samples of {digits.images.shape[1]}x{digits.images.shape[2]} pixels")
        print(f"   Features: {digits.data.shape[1]} (flattened pixel values)")
        print(f"   Classes: {len(digits.target_names)} digits (0-9)")
        print(f"   Class distribution:")
        
        unique, counts = np.unique(digits.target, return_counts=True)
        for digit, count in zip(unique, counts):
            print(f"     Digit {digit}: {count} samples")
        
        return {
            "images": digits.images,
            "data": digits.data,
            "target": digits.target,
            "target_names": digits.target_names
        }

    pipeline.add_stage(FunctionStage(
        name="load_data",
        func=load_digits_data,
        description="Load handwritten digits dataset"
    ))

    # Stage 2: Visualize sample digits
    def visualize_samples(context):
        """Display sample digits from each class."""
        images = context["load_data"]["images"]
        target = context["load_data"]["target"]
        
        print("\n👀 Sample digits visualization:")
        
        # Find one example of each digit
        sample_indices = []
        sample_digits = []
        for digit in range(10):
            idx = np.where(target == digit)[0][0]
            sample_indices.append(idx)
            sample_digits.append(images[idx])
        
        # Create a simple text visualization
        print("   Sample images (8x8 pixels, simplified view):")
        for i, (digit, image) in enumerate(zip(range(10), sample_digits)):
            print(f"   Digit {digit}:")
            # Simple ASCII representation (just show pattern)
            ascii_img = ""
            for row in image:
                row_str = ""
                for pixel in row:
                    if pixel > 8:  # Threshold for visibility
                        row_str += "█"
                    elif pixel > 4:
                        row_str += "▓"
                    elif pixel > 1:
                        row_str += "░"
                    else:
                        row_str += " "
                ascii_img += "     " + row_str + "\n"
            print(ascii_img)
            if i == 2:  # Show first 3 for brevity
                print("     ... (showing first 3 digits)")
                break
        
        return {
            "sample_indices": sample_indices,
            "sample_images": sample_digits
        }

    pipeline.add_stage(FunctionStage(
        name="visualize",
        func=visualize_samples,
        description="Visualize sample digits"
    ))

    # Stage 3: Preprocess and split data
    def preprocess_and_split(context):
        """Preprocess pixel data and split into train/test sets."""
        data = context["load_data"]["data"]
        target = context["load_data"]["target"]
        
        # Normalize pixel values to 0-1 range
        data_normalized = data / 16.0  # Original range is 0-16
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            data_normalized, target, test_size=0.2, random_state=42, stratify=target
        )
        
        print(f"\n📊 Data preprocessing:")
        print(f"   Pixel values normalized to [0, 1] range")
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        print(f"   Feature dimensions: {X_train.shape[1]} pixels")
        
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "data_normalized": data_normalized
        }

    pipeline.add_stage(FunctionStage(
        name="preprocess",
        func=preprocess_and_split,
        description="Normalize and split data"
    ))

    # Stage 4: Further scale for linear models
    def scale_features(context):
        """Apply StandardScaler for linear models."""
        X_train = context["preprocess"]["X_train"]
        X_test = context["preprocess"]["X_test"]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("\n🔧 Features scaled with StandardScaler for linear models")
        
        return {
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled,
            "scaler": scaler
        }

    pipeline.add_stage(FunctionStage(
        name="scale",
        func=scale_features,
        description="Scale features for linear models"
    ))

    # Stage 5: Train Logistic Regression
    def train_logistic_regression(context):
        """Train a Logistic Regression classifier."""
        X_train = context["scale"]["X_train_scaled"]
        y_train = context["preprocess"]["y_train"]
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        print("📊 Logistic Regression trained")
        
        return {"lr_model": model}

    pipeline.add_stage(FunctionStage(
        name="train_lr",
        func=train_logistic_regression,
        description="Train Logistic Regression"
    ))

    # Stage 6: Train SVM
    def train_svm(context):
        """Train an SVM classifier."""
        X_train = context["scale"]["X_train_scaled"]
        y_train = context["preprocess"]["y_train"]
        
        # Use RBF kernel with reasonable parameters for this dataset size
        model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        model.fit(X_train, y_train)
        
        print("🎯 SVM (RBF kernel) trained")
        
        return {"svm_model": model}

    pipeline.add_stage(FunctionStage(
        name="train_svm",
        func=train_svm,
        description="Train SVM classifier"
    ))

    # Stage 7: Train Random Forest
    def train_random_forest(context):
        """Train a Random Forest classifier."""
        X_train = context["preprocess"]["X_train"]  # Use normalized but not scaled data
        y_train = context["preprocess"]["y_train"]
        
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=20
        )
        model.fit(X_train, y_train)
        
        print("🌲 Random Forest trained")
        
        return {"rf_model": model}

    pipeline.add_stage(FunctionStage(
        name="train_rf",
        func=train_random_forest,
        description="Train Random Forest"
    ))

    # Stage 8: Evaluate all models
    def evaluate_models(context):
        """Evaluate all models and compare performance."""
        X_test_scaled = context["scale"]["X_test_scaled"]
        X_test = context["preprocess"]["X_test"]
        y_test = context["preprocess"]["y_test"]
        
        models = {
            "Logistic Regression": (context["train_lr"]["lr_model"], X_test_scaled),
            "SVM": (context["train_svm"]["svm_model"], X_test_scaled),
            "Random Forest": (context["train_rf"]["rf_model"], X_test)
        }
        
        results = {}
        
        print("\n🎯 Model Evaluation Results")
        print("-" * 40)
        
        best_accuracy = 0
        best_model_name = ""
        best_model = None
        best_predictions = None
        
        for name, (model, X_eval) in models.items():
            predictions = model.predict(X_eval)
            accuracy = accuracy_score(y_test, predictions)
            
            results[name] = {
                "predictions": predictions,
                "accuracy": accuracy
            }
            
            print(f"\n{name}:")
            print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
                best_model = model
                best_predictions = predictions
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        
        # Detailed classification report
        print(f"\n📋 Detailed Report for {best_model_name}:")
        target_names = [str(i) for i in range(10)]
        report = classification_report(y_test, best_predictions, target_names=target_names)
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, best_predictions)
        print("\n📊 Confusion Matrix:")
        print("Predicted →")
        print("Actual ↓  ", end="")
        print("  ".join([f"{i:2}" for i in range(10)]))
        for i, row in enumerate(cm):
            print(f"    {i}     ", end="")
            print("  ".join([f"{val:2}" for val in row]))
        
        return {
            "results": results,
            "best_model": best_model,
            "best_model_name": best_model_name,
            "best_accuracy": best_accuracy,
            "best_predictions": best_predictions,
            "confusion_matrix": cm,
            "y_test": y_test
        }

    pipeline.add_stage(FunctionStage(
        name="evaluate",
        func=evaluate_models,
        description="Evaluate and compare models"
    ))

    # Stage 9: Test individual predictions
    def test_individual_predictions(context):
        """Test predictions on individual digits."""
        best_model = context["evaluate"]["best_model"]
        best_model_name = context["evaluate"]["best_model_name"]
        images = context["load_data"]["images"]
        y_test = context["evaluate"]["y_test"]
        best_predictions = context["evaluate"]["best_predictions"]
        scaler = context["scale"]["scaler"]
        data_normalized = context["preprocess"]["data_normalized"]
        
        print(f"\n🔮 Individual Predictions using {best_model_name}:")
        print("-" * 50)
        
        # Find some interesting cases: correct and incorrect predictions
        test_indices = context["preprocess"]["X_test"]
        
        # Get a few test samples
        sample_indices = [0, 5, 10, 15, 20]  # Fixed indices for reproducibility
        
        for i, idx in enumerate(sample_indices[:5]):
            actual = y_test[idx]
            predicted = best_predictions[idx]
            
            # Get the original image for visualization
            test_sample = context["preprocess"]["X_test"][idx]
            
            # Reconstruct image (8x8) from flattened data
            image_2d = test_sample.reshape(8, 8)
            
            status = "✅ CORRECT" if actual == predicted else "❌ WRONG"
            print(f"\nSample {i+1}: {status}")
            print(f"  Actual: {actual}, Predicted: {predicted}")
            print(f"  Image (8x8 pixels):")
            
            # ASCII representation
            for row in image_2d:
                row_str = "    "
                for pixel in row:
                    if pixel > 0.5:
                        row_str += "██"
                    elif pixel > 0.3:
                        row_str += "▓▓"
                    elif pixel > 0.1:
                        row_str += "░░"
                    else:
                        row_str += "  "
                print(row_str)
        
        # Calculate error analysis
        errors = np.where(y_test != best_predictions)[0]
        error_rate = len(errors) / len(y_test)
        
        print(f"\n📊 Error Analysis:")
        print(f"   Total test samples: {len(y_test)}")
        print(f"   Correct predictions: {len(y_test) - len(errors)}")
        print(f"   Errors: {len(errors)}")
        print(f"   Error rate: {error_rate:.4f} ({error_rate*100:.2f}%)")
        
        if len(errors) > 0:
            print(f"   Most confused digits:")
            # Find most common errors
            error_pairs = [(y_test[i], best_predictions[i]) for i in errors]
            from collections import Counter
            most_common = Counter(error_pairs).most_common(3)
            for (actual, predicted), count in most_common:
                print(f"     {actual} → {predicted}: {count} times")
        
        return {
            "error_rate": error_rate,
            "total_errors": len(errors),
            "sample_results": [(y_test[i], best_predictions[i]) for i in sample_indices[:5]]
        }

    pipeline.add_stage(FunctionStage(
        name="test_individual",
        func=test_individual_predictions,
        description="Test individual digit predictions"
    ))

    # Define pipeline dependencies
    pipeline.add_dependency("load_data", "visualize")
    pipeline.add_dependency("visualize", "preprocess")
    pipeline.add_dependency("preprocess", "scale")
    pipeline.add_dependency("scale", "train_lr")
    pipeline.add_dependency("scale", "train_svm")
    pipeline.add_dependency("preprocess", "train_rf")
    pipeline.add_dependency("train_lr", "evaluate")
    pipeline.add_dependency("train_svm", "evaluate")
    pipeline.add_dependency("train_rf", "evaluate")
    pipeline.add_dependency("evaluate", "test_individual")

    # Show pipeline structure
    print("\n🔄 Pipeline Structure:")
    print(pipeline.visualize())
    print()

    # Run the pipeline
    print("🚀 Running pipeline...")
    results = pipeline.run()

    # Final summary
    print("\n" + "="*55)
    print("✅ Pipeline completed successfully!")
    print(f"🏆 Best model: {results['evaluate']['best_model_name']}")
    print(f"🎯 Accuracy: {results['evaluate']['best_accuracy']:.4f} ({results['evaluate']['best_accuracy']*100:.2f}%)")
    print(f"❌ Error rate: {results['test_individual']['error_rate']:.4f} ({results['test_individual']['error_rate']*100:.2f}%)")
    print("="*55)

    return results


if __name__ == "__main__":
    results = main()