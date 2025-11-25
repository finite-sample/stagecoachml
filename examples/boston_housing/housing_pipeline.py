"""Boston Housing regression pipeline using StagecoachML.

This example demonstrates a regression workflow:
1. Load Boston housing dataset from sklearn
2. Explore data characteristics
3. Split into train/test sets
4. Train multiple regression models
5. Compare performance with different metrics
6. Make predictions with confidence intervals

Run with: python examples/boston_housing/housing_pipeline.py
"""

import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import warnings
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from stagecoachml import Pipeline
from stagecoachml.stage import FunctionStage

# Suppress sklearn warnings about Boston housing dataset
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    """Run the complete California housing regression pipeline."""
    print("🏠 California Housing Price Prediction with StagecoachML")
    print("=" * 55)

    # Create pipeline
    pipeline = Pipeline(name="california_housing_regression")

    # Stage 1: Load California housing dataset
    def load_housing_data(context):
        """Load the California housing dataset from sklearn."""
        california = fetch_california_housing()
        print(f"🏠 Loaded California housing dataset: {california.data.shape[0]} samples, {california.data.shape[1]} features")
        print(f"📊 Target: Median house value (in hundreds of thousands of dollars)")
        print(f"📈 Price range: ${california.target.min():.1f} - ${california.target.max():.1f} (hundreds of thousands)")
        print(f"📊 Average price: ${california.target.mean():.1f} (hundreds of thousands)")
        
        return {
            "X": california.data,
            "y": california.target,
            "feature_names": california.feature_names,
            "description": california.DESCR
        }

    pipeline.add_stage(FunctionStage(
        name="load_data",
        func=load_housing_data,
        description="Load Boston housing dataset"
    ))

    # Stage 2: Data exploration
    def explore_data(context):
        """Explore dataset characteristics."""
        X = context["load_data"]["X"]
        y = context["load_data"]["y"]
        feature_names = context["load_data"]["feature_names"]
        
        print("\n📊 Data Exploration:")
        print("-" * 20)
        print(f"Features: {len(feature_names)}")
        print(f"Samples: {len(y)}")
        
        # Feature statistics
        feature_stats = {}
        for i, name in enumerate(feature_names):
            feature_data = X[:, i]
            stats = {
                "mean": feature_data.mean(),
                "std": feature_data.std(),
                "min": feature_data.min(),
                "max": feature_data.max()
            }
            feature_stats[name] = stats
        
        # Show some key features
        key_features = ["MedInc", "HouseAge", "AveRooms", "Population"]
        for feat in key_features:
            if feat in feature_names:
                idx = list(feature_names).index(feat)
                stats = feature_stats[feat]
                print(f"{feat}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
        
        return {
            "feature_stats": feature_stats,
            "n_features": len(feature_names),
            "n_samples": len(y)
        }

    pipeline.add_stage(FunctionStage(
        name="explore",
        func=explore_data,
        description="Explore data characteristics"
    ))

    # Stage 3: Split data
    def split_data(context):
        """Split data into training and testing sets."""
        X = context["load_data"]["X"]
        y = context["load_data"]["y"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\n📊 Data Split:")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Train price range: ${y_train.min():.1f} - ${y_train.max():.1f} (hundreds of thousands)")
        print(f"Test price range: ${y_test.min():.1f} - ${y_test.max():.1f} (hundreds of thousands)")
        
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

    # Stage 4: Scale features
    def scale_features(context):
        """Scale features using StandardScaler."""
        X_train = context["split"]["X_train"]
        X_test = context["split"]["X_test"]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("\n🔧 Features scaled using StandardScaler")
        
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

    # Stage 5: Train Linear Regression
    def train_linear_regression(context):
        """Train a Linear Regression model."""
        X_train = context["scale"]["X_train_scaled"]
        y_train = context["split"]["y_train"]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        print("📈 Linear Regression trained")
        
        return {"lr_model": model}

    pipeline.add_stage(FunctionStage(
        name="train_lr",
        func=train_linear_regression,
        description="Train Linear Regression"
    ))

    # Stage 6: Train Ridge Regression
    def train_ridge_regression(context):
        """Train a Ridge Regression model."""
        X_train = context["scale"]["X_train_scaled"]
        y_train = context["split"]["y_train"]
        
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)
        
        print("📈 Ridge Regression trained")
        
        return {"ridge_model": model}

    pipeline.add_stage(FunctionStage(
        name="train_ridge",
        func=train_ridge_regression,
        description="Train Ridge Regression"
    ))

    # Stage 7: Train Random Forest
    def train_random_forest(context):
        """Train a Random Forest Regressor."""
        X_train = context["split"]["X_train"]  # Use original features for tree-based model
        y_train = context["split"]["y_train"]
        
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        model.fit(X_train, y_train)
        
        # Feature importance
        feature_names = context["load_data"]["feature_names"]
        importances = model.feature_importances_
        feature_importance = dict(zip(feature_names, importances))
        
        print("🌲 Random Forest trained")
        print("   Top 3 important features:")
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        for feat, imp in top_features:
            print(f"   {feat}: {imp:.3f}")
        
        return {
            "rf_model": model,
            "feature_importance": feature_importance
        }

    pipeline.add_stage(FunctionStage(
        name="train_rf",
        func=train_random_forest,
        description="Train Random Forest Regressor"
    ))

    # Stage 8: Train Gradient Boosting
    def train_gradient_boosting(context):
        """Train a Gradient Boosting Regressor."""
        X_train = context["split"]["X_train"]  # Use original features
        y_train = context["split"]["y_train"]
        
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        print("🚀 Gradient Boosting trained")
        
        return {"gb_model": model}

    pipeline.add_stage(FunctionStage(
        name="train_gb",
        func=train_gradient_boosting,
        description="Train Gradient Boosting Regressor"
    ))

    # Stage 9: Evaluate all models
    def evaluate_models(context):
        """Evaluate all models and compare performance."""
        X_test_scaled = context["scale"]["X_test_scaled"]
        X_test = context["split"]["X_test"]
        y_test = context["split"]["y_test"]
        
        models = {
            "Linear Regression": context["train_lr"]["lr_model"],
            "Ridge Regression": context["train_ridge"]["ridge_model"],
            "Random Forest": context["train_rf"]["rf_model"],
            "Gradient Boosting": context["train_gb"]["gb_model"]
        }
        
        results = {}
        
        print("\n🎯 Model Evaluation Results")
        print("-" * 40)
        
        for name, model in models.items():
            # Use scaled features for linear models, original for tree-based
            X_eval = X_test_scaled if "Regression" in name else X_test
            
            predictions = model.predict(X_eval)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            results[name] = {
                "predictions": predictions,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
            
            print(f"\n{name}:")
            print(f"  RMSE: ${rmse:.2f} (hundreds of thousands)")
            print(f"  MAE:  ${mae:.2f} (hundreds of thousands)")
            print(f"  R²:   {r2:.4f}")
        
        # Find best model (highest R²)
        best_model_name = max(results.keys(), key=lambda k: results[k]["r2"])
        best_model = models[best_model_name]
        best_metrics = results[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   R² Score: {best_metrics['r2']:.4f}")
        print(f"   RMSE: ${best_metrics['rmse']:.2f} (hundreds of thousands)")
        
        return {
            "results": results,
            "best_model": best_model,
            "best_model_name": best_model_name,
            "best_metrics": best_metrics
        }

    pipeline.add_stage(FunctionStage(
        name="evaluate",
        func=evaluate_models,
        description="Evaluate and compare all models"
    ))

    # Stage 10: Make predictions on sample houses
    def make_sample_predictions(context):
        """Make predictions on sample house characteristics."""
        best_model = context["evaluate"]["best_model"]
        best_model_name = context["evaluate"]["best_model_name"]
        scaler = context["scale"]["scaler"]
        feature_names = context["load_data"]["feature_names"]
        
        # Sample house data (representative examples for California)
        # [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
        sample_houses = np.array([
            [5.0, 10.0, 6.5, 1.2, 3500.0, 3.0, 34.0, -118.0],  # Average LA area home
            [8.0, 5.0, 7.5, 1.3, 2000.0, 2.5, 37.8, -122.4],   # Nice SF area home
            [2.0, 40.0, 4.5, 1.5, 5000.0, 4.0, 32.7, -117.2],  # Older San Diego home
        ])
        
        house_descriptions = [
            "Average Los Angeles area home",
            "Nice San Francisco area home (higher income, newer)",
            "Older San Diego home (lower income, higher occupancy)"
        ]
        
        print(f"\n🏡 Sample Predictions using {best_model_name}:")
        print("-" * 50)
        
        predictions = []
        for i, (house_data, description) in enumerate(zip(sample_houses, house_descriptions)):
            # Scale features if needed
            if "Regression" in best_model_name:
                house_scaled = scaler.transform(house_data.reshape(1, -1))
                prediction = best_model.predict(house_scaled)[0]
            else:
                prediction = best_model.predict(house_data.reshape(1, -1))[0]
            
            predictions.append(prediction)
            
            print(f"\nHouse {i+1}: {description}")
            print(f"  Key features:")
            print(f"    Median income (MedInc): ${house_data[0]:.1f}0k")
            print(f"    House age: {house_data[1]:.0f} years")
            print(f"    Avg rooms: {house_data[2]:.1f}")
            print(f"    Population: {house_data[4]:.0f}")
            print(f"  💰 Predicted price: ${prediction:.1f} (hundreds of thousands)")
        
        return {
            "sample_predictions": predictions,
            "sample_descriptions": house_descriptions
        }

    pipeline.add_stage(FunctionStage(
        name="predict",
        func=make_sample_predictions,
        description="Make predictions on sample houses"
    ))

    # Define pipeline dependencies
    pipeline.add_dependency("load_data", "explore")
    pipeline.add_dependency("explore", "split")
    pipeline.add_dependency("split", "scale")
    pipeline.add_dependency("scale", "train_lr")
    pipeline.add_dependency("scale", "train_ridge")
    pipeline.add_dependency("split", "train_rf")  # RF uses original features
    pipeline.add_dependency("split", "train_gb")  # GB uses original features
    pipeline.add_dependency("train_lr", "evaluate")
    pipeline.add_dependency("train_ridge", "evaluate")
    pipeline.add_dependency("train_rf", "evaluate")
    pipeline.add_dependency("train_gb", "evaluate")
    pipeline.add_dependency("evaluate", "predict")

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
    print(f"📊 R² Score: {results['evaluate']['best_metrics']['r2']:.4f}")
    print(f"💰 RMSE: ${results['evaluate']['best_metrics']['rmse']:.2f} (hundreds of thousands)")
    print("="*55)

    return results


if __name__ == "__main__":
    results = main()