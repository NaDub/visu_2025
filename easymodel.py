import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier, plot_tree


class EasyModel:
    """
    A class to automate model training, evaluation, and selection.

    This class provides utilities for:
    - Splitting training and testing data.
    - Standardize Training and Testing data by saving it on self.stand_params
    - Training and evaluating machine learning models.
    - Hyperparameter tuning using GridSearchCV.
    - Visualizing performance with ROC curves and confusion matrices.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param test_size: Fraction of data used for testing
    :param random_state: Random seed for reproducibility

    Example usage:
    ```python
    model_manager = EasyModel(X, y)
    
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier()
    }

    param_grids = {
        "Logistic Regression": {'C': [0.1, 1, 10]},
        "Random Forest": {'n_estimators': [10, 50, 100]}
    }

    results = model_manager.model_selection(models, param_grids)
    best_model = results["Random Forest"]["best_estimator"]
    
    model_manager.assess_model(best_model)
    model_manager.plot_roc_curves({name: res["best_estimator"] for name, res in results.items()})
    model_manager.plot_confusion_matrix(best_model)
    ```
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, 
                                                                                random_state=random_state)
        self.stand_params = {}

    def assess_model(self, estimator: ClassifierMixin):
        """
        Train, predict, and evaluate a model.
        
        :param estimator: Model of type ClassifierMixin
        """
        try:
            estimator.fit(self.X_train, self.y_train)
            y_pred = estimator.predict(self.X_test)
            score = accuracy_score(self.y_test, y_pred)

            print(f"Accuracy: {score:.4f}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
            
            return score
        except Exception as e:
            print(f"Error during model evaluation: {e}")
            return None

    def model_selection(self, models: dict, param_grids: dict, cv: int = 5):
        """
        Perform hyperparameter tuning using GridSearchCV for each model.
        
        :param models: Dictionary of models with their names as keys
        :param param_grids: Dictionary of hyperparameter grids corresponding to each model
        :param cv: Number of folds for cross-validation
        """
        results = {}

        for model_name, estimator in models.items():
            if model_name in param_grids:
                param_grid = param_grids[model_name]
                try:
                    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, 
                                               cv=cv, n_jobs=-1, verbose=1, scoring='accuracy')
                    grid_search.fit(self.X_train, self.y_train)
                    
                    results[model_name] = {
                        "best_estimator": grid_search.best_estimator_,
                        "best_params": grid_search.best_params_,
                        "best_score": grid_search.best_score_
                    }
                    
                    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
                except Exception as e:
                    print(f"Error during model selection for {model_name}: {e}")
        
        return results
    
    def plot_roc_curves(self, models: dict):
        """
        Generate ROC curves for given models.
        
        :param models: Dictionary of trained models with their names as keys
        """
        plt.figure(figsize=(10, 8))
        best_models = {model: data['best_estimator'] for model, data in models.items()}
        for model_name, model in best_models.items():
            try:
                y_prob = model.predict_proba(self.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(self.y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
            except Exception as e:
                print(f"Error generating ROC curve for {model_name}: {e}")
        
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

    def plot_confusion_matrix(self, models: ClassifierMixin):
        """
        Generate and display the confusion matrix for a given model.
        
        :param model: Trained model of type ClassifierMixin
        """
        best_models = {model: data['best_estimator'] for model, data in models.items()}
        for model_name, model in best_models.items():
            try:
                y_pred = model.predict(self.X_test)
                cm = confusion_matrix(self.y_test, y_pred)
                
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title(f'Confusion Matrix of {model_name}')
                plt.show()
            except Exception as e:
                print(f"Error generating confusion matrix: {e}")
    
    def data_profit_curve(self, models: dict, profit: float, cost: float):
        """
        Generate profit curve data for given models.
        
        :param models: Dictionary of trained models with their names as keys
        :param profit: Profit for a correct positive prediction
        :param cost: Cost for a false positive prediction
        :return: Dictionary with model names as keys and their respective cumulative profit data as values
        """
        profit_data = {}
        best_models = {model: data['best_estimator'] for model, data in models.items()}
        
        for model_name, model in best_models.items():
            try:
                y_prob = model.predict_proba(self.X_test)[:, 1]
                sorted_indices = np.argsort(y_prob)[::-1]
                y_sorted = self.y_test.iloc[sorted_indices]
                
                profits = np.cumsum(y_sorted * profit - (1 - y_sorted) * cost)
                profit_data[model_name] = profits
            except Exception as e:
                print(f"Error generating profit data for {model_name}: {e}")
        
        return profit_data
    
    def data_best_profit(self, profit_data):
        """
        Get the maximum profit value for each model.

        :param profit_data: Dictionary with model names as keys and their respective cumulative profit data as values
        :return: Dictionary with model names as keys and their maximum profit values as values
        """
        self.best_profits = {model_name: np.max(profits) for model_name, profits in profit_data.items()}
        return self.best_profits

    
    def plot_profit_curve(self, models: dict, profit: float, cost: float):
        """
        Generate and display the profit curve for given models.
        
        :param models: Dictionary of trained models with their names as keys
        :param profit: Profit for a correct positive prediction
        :param cost: Cost for a false positive prediction
        """
        profit_data = self.data_profit_curve(models, profit, cost)
        best_profits = self.data_best_profit(profit_data)
        
        plt.figure(figsize=(10, 6))
        
        for model_name, profits in profit_data.items():
            tot_instances = len(profits)
            percent_instances = np.linspace(1 / tot_instances, 1, tot_instances) * 100
            plt.plot(percent_instances, profits, lw=2, label=f'{model_name}')
        
        for model_name, best_profit in best_profits.items():
            plt.axvline(x=best_profit, color='black', linestyle='--', label=f'Optimal Calls: {model_name}')

        plt.xlabel('Number of Instances')
        plt.ylabel('Cumulative Profit')
        plt.title('Profit Curve for Multiple Models')
        plt.legend(loc='upper right')
        plt.grid()
        plt.show()

    def mean_profit_curve(self, models: dict, profit: float, cost: float):
        """
        Generate and display the profit curve for given models.
        
        :param models: Dictionary of trained models with their names as keys
        :param profit: Profit for a correct positive prediction
        :param cost: Cost for a false positive prediction
        """
        profit_data = self.data_profit_curve(models, profit, cost)
        
        plt.figure(figsize=(10, 6))

        for model_name, profits in profit_data.items():
            tot_instances = len(profits)
            max_index = int(0.3 * tot_instances)
            percent_instances = np.linspace(1 / tot_instances, 1, tot_instances) * 100
            mean_profits = profits / np.arange(1, tot_instances + 1)  # Calcul du profit moyen

            plt.plot(percent_instances[:max_index], mean_profits[:max_index], lw=2, label=f'{model_name}')
        
        plt.xlabel('Percentage of Instances Processed')
        plt.ylabel('Mean Profit per Instance')
        plt.title('Mean Profit Curve for Multiple Models')
        plt.legend(loc='upper right')
        plt.grid()
        plt.show()

    def standardize_train_data(self, columns):
        """
        Standardize specified columns in a DataFrame using their mean and standard deviation.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            columns (list): A list of column names to standardize.

        Returns:
            pd.DataFrame: A copy of the DataFrame with the specified columns standardized.
            dict: A dictionary containing the mean and standard deviation for each standardized column.
        """
        scaler = StandardScaler()
        self.X_train[columns] = scaler.fit_transform(self.X_train[columns])
        self.stand_params = {
            col: {"mean": scaler.mean_[i], "std": scaler.scale_[i]} 
            for i, col in enumerate(columns)
        }

    def standardize_test_data(self, columns):
        """
        Standardize specified columns in a test DataFrame using the provided mean and standard deviation.

        Args:
            df (pd.DataFrame): The test DataFrame to be standardized.
            columns (list): A list of column names to standardize.
            stats (dict): A dictionary containing the mean and standard deviation for each column 
                        (from the training dataset).

        Returns:
            pd.DataFrame: A copy of the test DataFrame with the specified columns standardized.
        """
        for col in columns:
            if col not in self.stand_params:
                raise ValueError(f"Statistics for column '{col}' not found in the provided stats.")
            if col not in self.X_test.columns:
                raise ValueError(f"Column '{col}' not found in the test DataFrame.")
            
            mean = self.stand_params[col]["mean"]
            std = self.stand_params[col]["std"]
            
            self.X_test[col] = (self.X_test[col] - mean) / std

    def plot_feature_importance(self, model, top_n=40):
        """
        Plots the feature importance for the given model.

        :param model: Trained model (must have feature_importances_ attribute)
        :param top_n: Number of top features to display (default=20)
        """
        if not hasattr(model, "feature_importances_"):
            raise ValueError("The provided model does not have feature_importances_ attribute.")

        # Importance des variables
        feature_importances = model.feature_importances_

        # Vérifier que les noms des colonnes existent dans l'objet
        if hasattr(self, "X_train"):
            feature_names = self.X_train.columns
        else:
            raise ValueError("Feature names could not be determined.")

        # Création du DataFrame trié
        important_features = pd.Series(feature_importances, index=feature_names).nlargest(top_n)

        # Affichage du graphique
        plt.figure(figsize=(10, 6))
        important_features.plot(kind='barh', color='royalblue')
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(f"Feature importance - {type(model).__name__}")
        plt.gca().invert_yaxis()  # Inverser l'ordre pour meilleure lisibilité
        plt.show()

    def plot_decision_tree(self, model, max_depth=None):
        """
        Plots the decision tree structure for a trained DecisionTreeClassifier.

        :param model: Trained DecisionTreeClassifier model.
        :param max_depth: Maximum depth to display in the tree (default=None, shows full tree).
        """
        if not isinstance(model, DecisionTreeClassifier):
            raise ValueError("The provided model is not a DecisionTreeClassifier.")

        # Création du graphique
        plt.figure(figsize=(41, 12))
        plot_tree(model, feature_names=self.X_train.columns, class_names=['Non Réponse', 'Réponse'],
                  filled=True, rounded=True, fontsize=8, max_depth=max_depth)
        plt.title(f"Decision Tree - {type(model).__name__}")
        plt.show()


# First creating a class to test easily different models.
# /!\/!\ This class was partially reused from part of previous projects and improved for iterative enhancement in future projects.
# /!\/!\ So the doc string is not up-to-date /!\/!\

class EasyModelOld:
    """
    A class to automate model training, evaluation, and selection.

    This class provides utilities for:
    - Splitting training and testing data.
    - Standardize Training and Testing data by saving it on self.stand_params
    - Training and evaluating machine learning models.
    - Hyperparameter tuning using GridSearchCV.
    - Visualizing performance with ROC curves and confusion matrices.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param test_size: Fraction of data used for testing
    :param random_state: Random seed for reproducibility

    Example usage:
    ```python
    model_manager = EasyModel(X, y)
    
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier()
    }

    param_grids = {
        "Logistic Regression": {'C': [0.1, 1, 10]},
        "Random Forest": {'n_estimators': [10, 50, 100]}
    }

    results = model_manager.model_selection(models, param_grids)
    best_model = results["Random Forest"]["best_estimator"]
    
    model_manager.assess_model(best_model)
    model_manager.plot_roc_curves({name: res["best_estimator"] for name, res in results.items()})
    model_manager.plot_confusion_matrix(best_model)
    ```
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, 
                                                                                random_state=random_state)
        self.stand_params = {}

    def assess_model(self, estimator: ClassifierMixin):
        """
        Train, predict, and evaluate a model.
        
        :param estimator: Model of type ClassifierMixin
        """
        try:
            estimator.fit(self.X_train, self.y_train)
            y_pred = estimator.predict(self.X_test)
            score = accuracy_score(self.y_test, y_pred)

            print(f"Accuracy: {score:.4f}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
            
            return score
        except Exception as e:
            print(f"Error during model evaluation: {e}")
            return None

    def model_selection(self, models: dict, param_grids: dict, cv: int = 5):
        """
        Perform hyperparameter tuning using GridSearchCV for each model.
        
        :param models: Dictionary of models with their names as keys
        :param param_grids: Dictionary of hyperparameter grids corresponding to each model
        :param cv: Number of folds for cross-validation
        """
        results = {}

        for model_name, estimator in models.items():
            if model_name in param_grids:
                param_grid = param_grids[model_name]
                try:
                    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, 
                                               cv=cv, n_jobs=-1, verbose=1, scoring='accuracy')
                    grid_search.fit(self.X_train, self.y_train)
                    
                    results[model_name] = {
                        "best_estimator": grid_search.best_estimator_,
                        "best_params": grid_search.best_params_,
                        "best_score": grid_search.best_score_
                    }
                    
                    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
                except Exception as e:
                    print(f"Error during model selection for {model_name}: {e}")
        
        return results
    
    def plot_roc_curves(self, models: dict):
        """
        Generate ROC curves for given models.
        
        :param models: Dictionary of trained models with their names as keys
        """
        plt.figure(figsize=(10, 8))
        best_models = {model: data['best_estimator'] for model, data in models.items()}
        for model_name, model in best_models.items():
            try:
                y_prob = model.predict_proba(self.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(self.y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
            except Exception as e:
                print(f"Error generating ROC curve for {model_name}: {e}")
        
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

    def plot_confusion_matrix(self, models: ClassifierMixin):
        """
        Generate and display the confusion matrix for a given model.
        
        :param model: Trained model of type ClassifierMixin
        """
        best_models = {model: data['best_estimator'] for model, data in models.items()}
        for model_name, model in best_models.items():
            try:
                y_pred = model.predict(self.X_test)
                cm = confusion_matrix(self.y_test, y_pred)
                
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title(f'Confusion Matrix of {model_name}')
                plt.show()
            except Exception as e:
                print(f"Error generating confusion matrix: {e}")
    
    def data_profit_curve(self, models: dict, profit: float, cost: float):
        """
        Generate profit curve data for given models.
        
        :param models: Dictionary of trained models with their names as keys
        :param profit: Profit for a correct positive prediction
        :param cost: Cost for a false positive prediction
        :return: Dictionary with model names as keys and their respective cumulative profit data as values
        """
        profit_data = {}
        best_models = {model: data['best_estimator'] for model, data in models.items()}
        
        for model_name, model in best_models.items():
            try:
                y_prob = model.predict_proba(self.X_test)[:, 1]
                sorted_indices = np.argsort(y_prob)[::-1]
                y_sorted = self.y_test.iloc[sorted_indices]
                
                profits = np.cumsum(y_sorted * profit - (1 - y_sorted) * cost)
                profit_data[model_name] = profits
            except Exception as e:
                print(f"Error generating profit data for {model_name}: {e}")
        
        return profit_data
    
    def plot_profit_curve(self, models: dict, profit: float, cost: float):
        """
        Generate and display the profit curve for given models.
        
        :param models: Dictionary of trained models with their names as keys
        :param profit: Profit for a correct positive prediction
        :param cost: Cost for a false positive prediction
        """
        profit_data = self.data_profit_curve(models, profit, cost)
        
        plt.figure(figsize=(10, 6))
        
        for model_name, profits in profit_data.items():
            tot_instances = len(profits)
            percent_instances = np.linspace(1 / tot_instances, 1, tot_instances) * 100
            plt.plot(percent_instances, profits, lw=2, label=f'{model_name}')

        plt.autoscale(enable=True, axis='both', tight=None)
        plt.xlabel('Number of Instances')
        plt.ylabel('Cumulative Profit')
        plt.title('Profit Curve for Multiple Models')
        plt.legend(loc='upper right')
        plt.grid()
        plt.show()

    def mean_profit_curve(self, models: dict, profit: float, cost: float):
        """
        Generate and display the profit curve for given models.
        
        :param models: Dictionary of trained models with their names as keys
        :param profit: Profit for a correct positive prediction
        :param cost: Cost for a false positive prediction
        """
        profit_data = self.data_profit_curve(models, profit, cost)
        
        plt.figure(figsize=(10, 6))

        for model_name, profits in profit_data.items():
            tot_instances = len(profits)
            max_index = int(0.3 * tot_instances)
            percent_instances = np.linspace(1 / tot_instances, 1, tot_instances) * 100
            mean_profits = profits / np.arange(1, tot_instances + 1)  # Calcul du profit moyen

            plt.plot(percent_instances[:max_index], mean_profits[:max_index], lw=2, label=f'{model_name}')
        
        plt.xlabel('Percentage of Instances Processed')
        plt.ylabel('Mean Profit per Instance')
        plt.title('Mean Profit Curve for Multiple Models')
        plt.legend(loc='upper right')
        plt.grid()
        plt.show()

    def standardize_train_data(self, columns):
        """
        Standardize specified columns in a DataFrame using their mean and standard deviation.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            columns (list): A list of column names to standardize.

        Returns:
            pd.DataFrame: A copy of the DataFrame with the specified columns standardized.
            dict: A dictionary containing the mean and standard deviation for each standardized column.
        """
        scaler = StandardScaler()
        self.X_train[columns] = scaler.fit_transform(self.X_train[columns])
        self.stand_params = {
            col: {"mean": scaler.mean_[i], "std": scaler.scale_[i]} 
            for i, col in enumerate(columns)
        }

    def standardize_test_data(self, columns):
        """
        Standardize specified columns in a test DataFrame using the provided mean and standard deviation.

        Args:
            df (pd.DataFrame): The test DataFrame to be standardized.
            columns (list): A list of column names to standardize.
            stats (dict): A dictionary containing the mean and standard deviation for each column 
                        (from the training dataset).

        Returns:
            pd.DataFrame: A copy of the test DataFrame with the specified columns standardized.
        """
        for col in columns:
            if col not in self.stand_params:
                raise ValueError(f"Statistics for column '{col}' not found in the provided stats.")
            if col not in self.X_test.columns:
                raise ValueError(f"Column '{col}' not found in the test DataFrame.")
            
            mean = self.stand_params[col]["mean"]
            std = self.stand_params[col]["std"]
            
            self.X_test[col] = (self.X_test[col] - mean) / std


