# stacking_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from scipy.stats import randint, uniform
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import shap
import warnings

warnings.filterwarnings('ignore')


class RobustDataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_columns = None
        self.fitted = False
        self.column_order = None

    def fit(self, X, y=None):
        self.column_order = X.columns.tolist()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = numeric_cols
        print(f"使用的特征列 ({len(self.feature_columns)}个): {self.feature_columns}")

        X_numeric = X[self.feature_columns]
        X_imputed = self.imputer.fit_transform(X_numeric)
        self.scaler.fit(X_imputed)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        if list(X.columns) != self.column_order:
            print("调整输入数据列顺序以匹配训练时顺序")
            X = X[self.column_order]

        missing_cols = set(self.feature_columns) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in input data: {missing_cols}")

        X_numeric = X[self.feature_columns]
        X_imputed = self.imputer.transform(X_numeric)
        X_scaled = self.scaler.transform(X_imputed)
        return X_scaled

    def get_feature_names(self):
        return self.feature_columns


class HyperparameterOptimizer:
    def __init__(self, cv=5, n_iter=20, random_state=42):
        self.cv = cv
        self.n_iter = n_iter
        self.random_state = random_state
        self.best_params_ = {}
        self.best_scores_ = {}
        self.search_results_ = {}

    def optimize_model(self, model, param_grid, X, y):
        try:
            random_search = RandomizedSearchCV(
                model, param_grid, n_iter=self.n_iter, cv=self.cv,
                scoring='r2', n_jobs=-1, random_state=self.random_state, verbose=0,
                return_train_score=True
            )
            random_search.fit(X, y)
            best_model = random_search.best_estimator_
            model_name = str(model).split('(')[0]
            self.best_params_[model_name] = random_search.best_params_
            self.best_scores_[model_name] = random_search.best_score_

            cv_results = random_search.cv_results_
            search_results_df = pd.DataFrame({
                'params': cv_results['params'],
                'mean_test_score': cv_results['mean_test_score'],
                'std_test_score': cv_results['std_test_score'],
                'mean_train_score': cv_results.get('mean_train_score', [np.nan] * len(cv_results['params'])),
                'std_train_score': cv_results.get('std_train_score', [np.nan] * len(cv_results['params'])),
                'rank_test_score': cv_results['rank_test_score']
            })

            y_var = np.var(y)
            test_r2_scores = cv_results['mean_test_score']
            test_rmse_scores = np.sqrt((1 - test_r2_scores) * y_var)

            if 'mean_train_score' in cv_results:
                train_r2_scores = cv_results['mean_train_score']
                train_rmse_scores = np.sqrt((1 - train_r2_scores) * y_var)
            else:
                train_rmse_scores = [np.nan] * len(test_r2_scores)

            search_results_df['mean_test_rmse'] = test_rmse_scores
            search_results_df['mean_train_rmse'] = train_rmse_scores

            self.search_results_[model_name] = {
                'best_params': random_search.best_params_,
                'best_score': random_search.best_score_,
                'best_test_r2': random_search.best_score_,
                'best_test_rmse': np.sqrt((1 - random_search.best_score_) * y_var),
                'search_results': search_results_df,
                'param_grid': param_grid
            }

            print(f"最佳参数: {random_search.best_params_}")
            print(f"最佳交叉验证R²: {random_search.best_score_:.4f}")
            print(f"最佳交叉验证RMSE: {np.sqrt((1 - random_search.best_score_) * y_var):.4f}")

            return best_model
        except Exception as e:
            print(f"超参数优化失败: {e}")
            return model


class StackingEnsembleModel:
    def __init__(self, target_name, optimize_hyperparams=True):
        self.target_name = target_name
        self.preprocessor = RobustDataPreprocessor()
        self.base_models = []
        self.meta_model = None
        self.meta_scaler = StandardScaler()
        self.fitted = False
        self.optimize_hyperparams = optimize_hyperparams
        self.optimizer = HyperparameterOptimizer()
        self.train_predictions = None
        self.test_predictions = None
        self.base_models_performance = {}
        self.meta_model_performance = {}
        self.base_models_cv_scores = {}
        self.y_var = None

    def _get_base_model_param_grids(self):
        return {
            'rf': {
                'n_estimators': randint(50, 200),
                'max_depth': [3, 5, 7, None],
                'min_samples_split': randint(2, 10),
                'min_samples_leaf': randint(1, 4)
            },
            'xgb': {
                'n_estimators': randint(50, 200),
                'max_depth': randint(3, 7),
                'learning_rate': uniform(0.01, 0.2),
                'subsample': uniform(0.8, 0.2),
                'colsample_bytree': uniform(0.8, 0.2)
            },
            'gbr': {
                'n_estimators': randint(50, 200),
                'max_depth': randint(3, 7),
                'learning_rate': uniform(0.01, 0.1),
                'subsample': uniform(0.8, 0.2)
            },
            'svr': {
                'C': uniform(0.1, 100),
                'epsilon': uniform(0.01, 0.2),
                'gamma': ['scale', 'auto']
            },
            'lgb': {
                'n_estimators': randint(50, 200),
                'max_depth': randint(3, 7),
                'learning_rate': uniform(0.01, 0.1),
                'num_leaves': randint(31, 100),
                'subsample': uniform(0.8, 0.2)
            }
        }

    def _get_base_models(self):
        return [
            ('rf', RandomForestRegressor(random_state=42)),
            ('xgb', XGBRegressor(random_state=42)),
            ('gbr', GradientBoostingRegressor(random_state=42)),
            ('svr', SVR()),
            ('lgb', LGBMRegressor(random_state=42, verbose=-1))
        ]

    def _optimize_base_models(self, X, y):
        print(f"\n开始超参数优化...")
        optimized_models = []
        param_grids = self._get_base_model_param_grids()

        for name, model in self._get_base_models():
            if name in param_grids:
                print(f"\n优化 {name} 模型...")
                try:
                    optimized_model = self.optimizer.optimize_model(
                        model, param_grids[name], X, y
                    )
                    optimized_models.append((name, optimized_model))
                except Exception as e:
                    print(f"{name} 优化失败，使用默认参数: {e}")
                    optimized_models.append((name, model))
            else:
                optimized_models.append((name, model))

        return optimized_models

    def fit(self, X, y):
        self.y_var = np.var(y)
        X_processed = self.preprocessor.fit_transform(X, y)

        print(f"\n训练堆叠模型 - 目标: {self.target_name}")
        print(f"训练集形状: {X_processed.shape}")

        if self.optimize_hyperparams:
            base_models = self._optimize_base_models(X_processed, y)
        else:
            base_models = self._get_base_models()

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        meta_features = np.zeros((X_processed.shape[0], len(base_models)))

        self.base_models = []
        base_models_train_scores = {}

        for i, (name, model) in enumerate(base_models):
            print(f"\n训练基模型: {name}")
            model_predictions = np.zeros(X_processed.shape[0])
            fold_scores = []
            fold_models = []

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_processed)):
                X_train_fold, X_val_fold = X_processed[train_idx], X_processed[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

                model_clone = clone(model)
                model_clone.fit(X_train_fold, y_train_fold)
                fold_models.append(model_clone)

                y_val_pred = model_clone.predict(X_val_fold)
                model_predictions[val_idx] = y_val_pred

                fold_r2 = r2_score(y_val_fold, y_val_pred)
                fold_scores.append(fold_r2)

            meta_features[:, i] = model_predictions

            best_fold_idx = np.argmax(fold_scores)
            self.base_models.append((name, fold_models[best_fold_idx]))

            cv_mean = np.mean(fold_scores)
            cv_std = np.std(fold_scores)
            self.base_models_cv_scores[name] = {
                'mean_score': cv_mean,
                'std_score': cv_std,
                'fold_scores': fold_scores
            }

            y_train_pred = fold_models[best_fold_idx].predict(X_processed)
            train_r2 = r2_score(y, y_train_pred)
            train_rmse = np.sqrt(mean_squared_error(y, y_train_pred))
            train_mae = mean_absolute_error(y, y_train_pred)

            self.base_models_performance[name] = {
                'train_r2': train_r2,
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                'cv_mean': cv_mean,
                'cv_std': cv_std
            }

            print(f"{name} 平均交叉验证R²: {cv_mean:.4f} (±{cv_std:.4f})")
            print(f"{name} 训练集R²: {train_r2:.4f}")

        meta_features_scaled = self.meta_scaler.fit_transform(meta_features)

        self.meta_model = LinearRegression()
        self.meta_model.fit(meta_features_scaled, y)

        meta_r2 = self.meta_model.score(meta_features_scaled, y)
        meta_predictions = self.meta_model.predict(meta_features_scaled)
        meta_rmse = np.sqrt(mean_squared_error(y, meta_predictions))
        meta_mae = mean_absolute_error(y, meta_predictions)

        self.meta_model_performance['train'] = {
            'r2': meta_r2,
            'rmse': meta_rmse,
            'mae': meta_mae
        }

        print(f"\n元模型训练R²: {meta_r2:.4f}")
        print(f"元模型训练RMSE: {meta_rmse:.4f}")

        self.train_predictions = meta_predictions
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        X_processed = self.preprocessor.transform(X)
        base_preds = []
        for name, model in self.base_models:
            pred = model.predict(X_processed)
            base_preds.append(pred)

        base_preds_array = np.column_stack(base_preds)
        base_preds_scaled = self.meta_scaler.transform(base_preds_array)

        predictions = self.meta_model.predict(base_preds_scaled)
        self.test_predictions = predictions
        return predictions

    def evaluate_test_performance(self, X_test, y_test):
        if not self.fitted:
            raise ValueError("Model must be fitted before evaluation")

        y_pred = self.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)

        self.meta_model_performance['test'] = {
            'r2': test_r2,
            'rmse': test_rmse,
            'mae': test_mae
        }

        X_processed = self.preprocessor.transform(X_test)
        for name, model in self.base_models:
            y_base_pred = model.predict(X_processed)
            base_r2 = r2_score(y_test, y_base_pred)
            base_rmse = np.sqrt(mean_squared_error(y_test, y_base_pred))
            base_mae = mean_absolute_error(y_test, y_base_pred)

            if 'test' not in self.base_models_performance[name]:
                self.base_models_performance[name]['test'] = {}

            self.base_models_performance[name]['test'] = {
                'r2': base_r2,
                'rmse': base_rmse,
                'mae': base_mae
            }

        return {
            'stacked_test_r2': test_r2,
            'stacked_test_rmse': test_rmse,
            'stacked_test_mae': test_mae
        }

    def get_hyperparameter_optimization_results(self):
        return self.optimizer.search_results_

    def get_base_models_performance(self):
        return self.base_models_performance

    def get_meta_model_performance(self):
        return self.meta_model_performance

    def get_base_models_cv_scores(self):
        return self.base_models_cv_scores

    def get_shap_values(self, X):
        try:
            explainer = shap.Explainer(self.base_models[0][1])
            X_processed = self.preprocessor.transform(X)
            shap_values = explainer(X_processed)
            return shap_values
        except Exception as e:
            print(f"SHAP计算失败: {e}")
            return None

    def get_feature_importance_ranking(self, X):
        shap_values = self.get_shap_values(X)
        if shap_values is not None:
            feature_importance = np.abs(shap_values.values).mean(0)
            feature_names = self.preprocessor.get_feature_names()

            if len(feature_names) != len(feature_importance):
                print(f"警告: 特征名称数量({len(feature_names)})与SHAP值数量({len(feature_importance)})不一致")
                min_length = min(len(feature_names), len(feature_importance))
                feature_names = feature_names[:min_length]
                feature_importance = feature_importance[:min_length]

            importance_df = pd.DataFrame({
                '特征名称': feature_names,
                'SHAP重要性': feature_importance
            })
            importance_df = importance_df.sort_values('SHAP重要性', ascending=False)
            return importance_df, shap_values
        else:
            return self.get_traditional_feature_importance(), None

    def get_traditional_feature_importance(self):
        try:
            if hasattr(self.base_models[0][1], 'feature_importances_'):
                importance = self.base_models[0][1].feature_importances_
                feature_names = self.preprocessor.get_feature_names()

                if len(feature_names) != len(importance):
                    print(f"警告: 特征名称数量({len(feature_names)})与重要性值数量({len(importance)})不一致")
                    min_length = min(len(feature_names), len(importance))
                    feature_names = feature_names[:min_length]
                    importance = importance[:min_length]

                importance_df = pd.DataFrame({
                    '特征名称': feature_names,
                    '特征重要性': importance
                })
                importance_df = importance_df.sort_values('特征重要性', ascending=False)
                return importance_df
        except Exception as e:
            print(f"传统特征重要性计算失败: {e}")

        return None

    def get_feature_target_relationship_data(self, X, y):
        feature_names = self.preprocessor.get_feature_names()
        X_processed = self.preprocessor.transform(X)
        relationship_data = {}
        relationship_data['目标值'] = y.values

        for i, feature in enumerate(feature_names):
            if i < X_processed.shape[1]:
                relationship_data[feature] = X_processed[:, i]

        return pd.DataFrame(relationship_data)

    def get_all_predictions(self, X_train, y_train, X_test, y_test):
        train_pred = self.train_predictions
        test_pred = self.test_predictions if self.test_predictions is not None else self.predict(X_test)

        predictions_df = pd.DataFrame({
            '数据集类型': ['训练集'] * len(y_train) + ['测试集'] * len(y_test),
            '真实值': np.concatenate([y_train.values, y_test.values]),
            '预测值': np.concatenate([train_pred, test_pred])
        })
        return predictions_df

    def get_test_set_with_predictions(self, X_test, y_test):
        test_pred = self.test_predictions if self.test_predictions is not None else self.predict(X_test)

        test_data_with_predictions = X_test.copy()
        test_data_with_predictions['真实值'] = y_test.values
        test_data_with_predictions['预测值'] = test_pred
        test_data_with_predictions['绝对误差'] = np.abs(y_test.values - test_pred)
        test_data_with_predictions['相对误差(%)'] = (np.abs(y_test.values - test_pred) / y_test.values) * 100

        return test_data_with_predictions