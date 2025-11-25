"""Complete iris classification pipeline using StagecoachML.

This example demonstrates a full ML workflow:
1. Load iris dataset from sklearn
2. Split into train/test sets
3. Scale features
4. Train multiple models
5. Compare performance
6. Make predictions

Run with: python examples/iris_classification/iris_pipeline.py
"""

import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from stagecoachml import Pipeline
from stagecoachml.stage import FunctionStage


def main():
    """Run the complete iris classification pipeline."""
    print("🌸 Iris Classification Pipeline with StagecoachML")
    print("=" * 50)

    # Create pipeline
    pipeline = Pipeline(name="iris_classification")

    # Stage 1: Load iris dataset
    def load_iris_data(context):
        """Load the iris dataset from sklearn."""
        iris = load_iris()
        print(f"📊 Loaded iris dataset: {iris.data.shape[0]} samples, {iris.data.shape[1]} features")
        print(f"📋 Classes: {iris.target_names}")
        return {
            "X": iris.data,
            "y": iris.target,
            "feature_names": iris.feature_names,
            "target_names": iris.target_names
        }

    pipeline.add_stage(FunctionStage(
        name="load_data",
        func=load_iris_data,
        description="Load iris dataset from sklearn"
    ))

    # Stage 2: Split data
    def split_data(context):
        """Split data into training and testing sets."""
        X = context["load_data"]["X"]
        y = context["load_data"]["y"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"📊 Train set: {X_train.shape[0]} samples")
        print(f"📊 Test set: {X_test.shape[0]} samples")
        
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }

    pipeline.add_stage(FunctionStage(
        name="split",
        func=split_data,
        description="Split data into train/test sets"
    ))

    # Stage 3: Scale features
    def scale_features(context):
        """Scale features using StandardScaler."""
        X_train = context["split"]["X_train"]
        X_test = context["split"]["X_test"]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("🔧 Features scaled using StandardScaler")
        print(f"   Mean: {X_train_scaled.mean(axis=0)}")
        print(f"   Std: {X_train_scaled.std(axis=0)}")
        
        return {
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled,
            "scaler": scaler
        }

    pipeline.add_stage(FunctionStage(
        name="scale",
        func=scale_features,
        description="Scale features with StandardScaler"
    ))

    # Stage 4: Train Random Forest
    def train_random_forest(context):
        """Train a Random Forest classifier."""
        X_train = context["scale"]["X_train_scaled"]
        y_train = context["split"]["y_train"]
        
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=5
        )
        model.fit(X_train, y_train)
        
        # Feature importance
        feature_names = context["load_data"]["feature_names"]
        importances = model.feature_importances_
        feature_importance = dict(zip(feature_names, importances))
        
        print("🌲 Random Forest trained")
        print("   Feature importances:")
        for feat, imp in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"   {feat}: {imp:.3f}")
        
        return {
            "rf_model": model,
            "feature_importance": feature_importance
        }

    pipeline.add_stage(FunctionStage(
        name="train_rf",
        func=train_random_forest,
        description="Train Random Forest classifier"
    ))

    # Stage 5: Train Logistic Regression
    def train_logistic_regression(context):
        """Train a Logistic Regression classifier."""
        X_train = context["scale"]["X_train_scaled"]
        y_train = context["split"]["y_train"]
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        print("📊 Logistic Regression trained")
        
        return {"lr_model": model}

    pipeline.add_stage(FunctionStage(
        name="train_lr",
        func=train_logistic_regression,
        description="Train Logistic Regression classifier"
    ))

    # Stage 6: Evaluate models
    def evaluate_models(context):
        """Evaluate both models and compare performance."""
        X_test = context["scale"]["X_test_scaled"]
        y_test = context["split"]["y_test"]
        rf_model = context["train_rf"]["rf_model"]
        lr_model = context["train_lr"]["lr_model"]
        target_names = context["load_data"]["target_names"]
        
        # Random Forest predictions
        rf_predictions = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_predictions)
        
        # Logistic Regression predictions
        lr_predictions = lr_model.predict(X_test)
        lr_accuracy = accuracy_score(y_test, lr_predictions)
        
        print("\n🎯 Model Evaluation Results")
        print("-" * 30)
        print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
        print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
        
        # Best model
        best_model_name = "Random Forest" if rf_accuracy > lr_accuracy else "Logistic Regression"
        best_model = rf_model if rf_accuracy > lr_accuracy else lr_model
        best_predictions = rf_predictions if rf_accuracy > lr_accuracy else lr_predictions
        best_accuracy = max(rf_accuracy, lr_accuracy)
        
        print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        # Detailed classification report
        print("\n📋 Detailed Classification Report:")
        print(classification_report(y_test, best_predictions, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, best_predictions)
        print("\n📊 Confusion Matrix:")
        print(cm)
        
        return {
            "rf_accuracy": rf_accuracy,
            "lr_accuracy": lr_accuracy,
            "best_model": best_model,
            "best_model_name": best_model_name,
            "best_accuracy": best_accuracy,
            "best_predictions": best_predictions,
            "confusion_matrix": cm,
            "classification_report": classification_report(y_test, best_predictions, target_names=target_names)
        }

    pipeline.add_stage(FunctionStage(
        name="evaluate",
        func=evaluate_models,
        description="Evaluate and compare model performance"
    ))

    # Stage 7: Make sample predictions
    def make_sample_predictions(context):
        """Make predictions on some sample data points."""
        best_model = context["evaluate"]["best_model"]
        scaler = context["scale"]["scaler"]
        target_names = context["load_data"]["target_names"]
        feature_names = context["load_data"]["feature_names"]
        
        # Create sample data points
        sample_data = np.array([
            [5.1, 3.5, 1.4, 0.2],  # Should be setosa
            [6.2, 2.8, 4.8, 1.8],  # Should be virginica
            [5.9, 3.0, 4.2, 1.5],  # Should be versicolor
        ])
        
        # Scale the sample data
        sample_scaled = scaler.transform(sample_data)
        
        # Make predictions
        predictions = best_model.predict(sample_scaled)
        probabilities = best_model.predict_proba(sample_scaled) if hasattr(best_model, 'predict_proba') else None
        
        print("\n🔮 Sample Predictions:")
        print("-" * 40)
        for i, (sample, pred) in enumerate(zip(sample_data, predictions)):
            feature_str = ", ".join([f"{name}={val:.1f}" for name, val in zip(feature_names, sample)])
            predicted_class = target_names[pred]
            print(f"Sample {i+1}: [{feature_str}]")
            print(f"  → Predicted: {predicted_class}")
            
            if probabilities is not None:
                prob_str = ", ".join([f"{name}={prob:.3f}" for name, prob in zip(target_names, probabilities[i])])
                print(f"  → Probabilities: [{prob_str}]")
            print()
        
        return {
            "sample_predictions": predictions,
            "sample_probabilities": probabilities,
            "sample_data": sample_data
        }

    pipeline.add_stage(FunctionStage(
        name="predict",
        func=make_sample_predictions,
        description="Make predictions on sample data"
    ))

    # Define pipeline dependencies
    pipeline.add_dependency("load_data", "split")
    pipeline.add_dependency("split", "scale")
    pipeline.add_dependency("scale", "train_rf")
    pipeline.add_dependency("scale", "train_lr")
    pipeline.add_dependency("train_rf", "evaluate")
    pipeline.add_dependency("train_lr", "evaluate")
    pipeline.add_dependency("evaluate", "predict")

    # Show pipeline structure
    print("\n🔄 Pipeline Structure:")
    print(pipeline.visualize())
    print()

    # Run the pipeline
    print("🚀 Running pipeline...")
    results = pipeline.run()

    # Final summary
    print("\n" + "="*50)
    print("✅ Pipeline completed successfully!")
    print(f"🏆 Best model: {results['evaluate']['best_model_name']}")
    print(f"📊 Accuracy: {results['evaluate']['best_accuracy']:.4f}")
    print("="*50)

    return results


if __name__ == "__main__":
    results = main()