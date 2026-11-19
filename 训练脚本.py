import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin, clone
import shap
import os
import tkinter as tk
from tkinter import filedialog
import warnings
from sklearn.impute import SimpleImputer
from scipy.stats import randint, uniform
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib

# 关键：从独立模块导入三个类，保证 pkl 里的类路径可被预测脚本解析
from stacking_model import RobustDataPreprocessor, HyperparameterOptimizer, StackingEnsembleModel

warnings.filterwarnings('ignore')


# 文件选择对话框函数
def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="选择数据文件",
        filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
    )
    return file_path


# 读取数据的函数
def load_data(file_path):
    if file_path.endswith('.csv'):
        encodings = ['utf-8', 'gbk', 'latin-1', 'utf-8-sig']
        for encoding in encodings:
            try:
                data = pd.read_csv(file_path, encoding=encoding)
                print(f"成功使用 {encoding} 编码读取CSV文件")
                return data
            except UnicodeDecodeError:
                continue
        raise ValueError("无法使用任何常见编码读取CSV文件")
    elif file_path.endswith('.xlsx'):
        try:
            data = pd.read_excel(file_path)
            print("成功读取Excel文件")
            return data
        except Exception as e:
            print(f"读取Excel文件时出错: {e}")
            return None
    else:
        print("不支持的文件格式")
        return None


# 主程序
def main():
    print("请选择数据文件...")
    file_path = select_file()

    if not file_path:
        print("未选择文件，程序退出")
        return

    print(f"已选择文件: {file_path}")
    data = load_data(file_path)

    if data is None:
        print("读取数据失败，程序退出")
        return

    print(f"数据形状: {data.shape}")
    print("数据前5行:")
    print(data.head())

    # 定义输入特征列名（根据您的要求修改）
    input_features = {
        'biochemical_composition': ['Pro (%)', 'Lip (%)', 'CH (%)'],  # 生化组成
        'elemental_analysis': ['C (%)', 'H (%)', 'O (%)', 'N (%)', 'S (%)'],  # 元素分析
        'ash': ['Ash (%)'],  # 灰分
        'operating_conditions': ['Time (min)', 'T (℃)', 'Size (L)', 'Ratio']  # 操作条件
    }

    # 反应溶剂和分离溶剂（可能为分类变量）
    solvent_features = ['RS', 'SS']

    # 所有输入特征列表
    all_input_features = []
    for category, features in input_features.items():
        all_input_features.extend(features)
    all_input_features.extend(solvent_features)

    print(f"\n期望的输入特征: {all_input_features}")

    # 检查实际存在的输入特征
    available_input_features = [col for col in all_input_features if col in data.columns]
    missing_features = set(all_input_features) - set(available_input_features)

    if missing_features:
        print(f"警告: 以下输入特征在数据中不存在: {missing_features}")
        print("将继续使用存在的特征进行建模")

    print(f"实际使用的输入特征 ({len(available_input_features)}个): {available_input_features}")

    # 定义目标变量（生物油性质）- 根据您的要求修改
    target_columns = {
        'Yield_oil': ['Yield_oil (wt%)', '生物油产量', 'Oil_Yield'],
        'Oil_C': ['Oil_C (%)', '生物油C含量', 'C_oil'],
        'Oil_H': ['Oil_H (%)', '生物油H含量', 'H_oil'],
        'Oil_O': ['Oil_O (%)', '生物油O含量', 'O_oil'],
        'Oil_N': ['Oil_N (%)', '生物油N含量', 'N_oil'],
        'Oil_S': ['Oil_S (%)', '生物油S含量', 'S_oil'],
        'Oil_HHV': ['HHV (MJ/kg)', '生物油热值', 'HHV_oil']
    }

    # 识别实际存在的目标变量
    available_targets = {}
    target_name_to_original = {}  # 存储重命名后的目标名到原始列名的映射
    for target_name, possible_cols in target_columns.items():
        for col in possible_cols:
            if col in data.columns:
                available_targets[target_name] = col
                target_name_to_original[target_name] = col
                break

    if not available_targets:
        print("未找到任何目标变量，请检查数据列名")
        return

    print(f"\n找到的目标变量: {list(available_targets.keys())}")
    print("目标变量对应的实际列名:")
    for target_name, actual_col in available_targets.items():
        print(f"  {target_name}: {actual_col}")

    # 分离特征和目标
    all_target_cols = list(available_targets.values())
    X = data[available_input_features]  # 只使用存在的输入特征
    y = data[all_target_cols]
    y_renamed = y.rename(columns={v: k for k, v in available_targets.items()})

    # 处理分类变量（反应溶剂RS和分离溶剂SS）
    categorical_cols = [col for col in solvent_features if col in available_input_features]
    if categorical_cols:
        print(f"\n处理分类变量: {categorical_cols}")
        # 使用独热编码处理分类变量
        X = pd.get_dummies(X, columns=categorical_cols, prefix=categorical_cols)

    print(f"\n处理后的特征数量: {X.shape[1]}")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_renamed, test_size=0.15, random_state=42
    )

    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"目标变量数量: {len(available_targets)}")

    # 为每个目标变量训练堆叠模型
    models = {}
    results = {}
    shap_results = {}
    feature_relationships = {}
    all_predictions = {}
    test_set_predictions = {}
    hyperparameter_results = {}  # 存储超参数优化结果
    model_performance_results = {}  # 存储模型性能结果

    for target_name in available_targets.keys():
        print(f"\n{'=' * 60}")
        print(f"训练堆叠模型: {target_name}")
        print(f"{'=' * 60}")

        try:
            # 创建并训练堆叠模型（现在使用5折交叉验证）
            model = StackingEnsembleModel(target_name, optimize_hyperparams=True)
            model.fit(X_train, y_train[target_name])

            # ===================== 新增：保存模型供软件加载 =====================
            MODEL_DIR = "trained_models"
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump({
                "model": model,
                "feature_order": available_input_features,   # 独热前的原始输入列
                "categorical_cols": categorical_cols,        # RS/SS 列名
                "category_levels": {
                    c: sorted(data[c].dropna().astype(str).unique().tolist())
                    for c in categorical_cols
                },
                "target_name": target_name,
                "target_display": target_name,
            }, f"{MODEL_DIR}/{target_name}.pkl")
            print(f"模型已保存: {MODEL_DIR}/{target_name}.pkl")
            # ====================================================================

            # 评估测试集性能
            test_performance = model.evaluate_test_performance(X_test, y_test[target_name])

            # 存储超参数优化结果
            hyperparameter_results[target_name] = model.get_hyperparameter_optimization_results()

            # 存储模型性能结果
            model_performance_results[target_name] = {
                'base_models': model.get_base_models_performance(),
                'meta_model': model.get_meta_model_performance(),
                'base_models_cv': model.get_base_models_cv_scores()
            }

            # 预测
            y_pred = model.predict(X_test)

            # 计算评估指标
            mse = mean_squared_error(y_test[target_name], y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test[target_name], y_pred)
            r2 = r2_score(y_test[target_name], y_pred)

            results[target_name] = {
                'y_test': y_test[target_name],
                'y_pred': y_pred,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }

            models[target_name] = model

            print(f"\n测试集结果:")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R²: {r2:.4f}")

            # 输出各基模型性能
            print(f"\n各基模型性能:")
            base_performance = model.get_base_models_performance()
            for model_name, perf in base_performance.items():
                print(f"{model_name}: 训练R²={perf['train_r2']:.4f}, 测试R²={perf['test']['r2']:.4f}")

            # 获取所有预测值（训练集+测试集）
            print("收集所有预测值...")
            predictions_df = model.get_all_predictions(X_train, y_train[target_name], X_test, y_test[target_name])
            all_predictions[target_name] = predictions_df

            # 获取测试集所有数据和预测值
            print("收集测试集数据和预测值...")
            test_data_with_pred = model.get_test_set_with_predictions(X_test, y_test[target_name])
            test_set_predictions[target_name] = test_data_with_pred
            print(f"测试集数据形状: {test_data_with_pred.shape}")

            # 获取SHAP值分析排行
            print("计算特征SHAP值分析排行...")
            importance_df, shap_values = model.get_feature_importance_ranking(X_train)
            if importance_df is not None:
                shap_results[target_name] = importance_df
                print(f"\n{target_name} - 前10个重要特征:")
                print(importance_df.head(10).to_string(index=False))
            else:
                print(f"无法计算 {target_name} 的SHAP值")

            # 获取特征与目标变量完整关系数据 - 使用原始数据而不是重命名后的数据
            print("计算特征与目标变量完整关系数据...")
            original_target_col = target_name_to_original[target_name]
            relationship_df = model.get_feature_target_relationship_data(X, data[original_target_col])
            feature_relationships[target_name] = relationship_df
            print(f"\n{target_name} - 特征与目标变量关系数据形状: {relationship_df.shape}")

        except Exception as e:
            print(f"训练 {target_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 创建结果目录
    results_dir = "生物油性质预测分析结果（5折交叉验证）高斯数据增强10"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 保存各目标变量的完整数据
    for target_name in available_targets.keys():
        if target_name in results:
            # 保存所有预测值（训练集+测试集）
            if target_name in all_predictions:
                all_predictions[target_name].to_csv(
                    f'{results_dir}/{target_name}_全部预测值.csv',
                    index=False, encoding='utf-8-sig'
                )

            # 保存测试集所有数据和预测值
            if target_name in test_set_predictions:
                test_set_predictions[target_name].to_csv(
                    f'{results_dir}/{target_name}_测试集完整数据.csv',
                    index=False, encoding='utf-8-sig'
                )

            # 保存SHAP重要性排行
            if target_name in shap_results:
                shap_results[target_name].to_csv(
                    f'{results_dir}/{target_name}_特征SHAP重要性排行.csv',
                    index=False, encoding='utf-8-sig'
                )

            # 保存特征与目标变量完整关系数据
            if target_name in feature_relationships:
                feature_relationships[target_name].to_csv(
                    f'{results_dir}/{target_name}_特征目标完整关系数据.csv',
                    index=False, encoding='utf-8-sig'
                )

    # 保存超参数优化结果
    hyperparameter_dir = f"{results_dir}/超参数优化结果"
    if not os.path.exists(hyperparameter_dir):
        os.makedirs(hyperparameter_dir)

    for target_name, hp_results in hyperparameter_results.items():
        for model_name, result in hp_results.items():
            # 保存最佳参数和性能
            best_params_df = pd.DataFrame([{
                'model': model_name,
                'target': target_name,
                'best_params': str(result['best_params']),
                'best_test_r2': result['best_test_r2'],
                'best_test_rmse': result['best_test_rmse']
            }])
            best_params_df.to_csv(
                f'{hyperparameter_dir}/{target_name}_{model_name}_最佳参数和性能.csv',
                index=False, encoding='utf-8-sig'
            )

            # 保存所有超参数组合的详细性能数据（包含R2和RMSE）
            search_results = result['search_results'].copy()
            search_results['model'] = model_name
            search_results['target'] = target_name

            # 展开params列，将每个参数作为单独的列
            params_df = pd.json_normalize(search_results['params'])
            search_results_expanded = pd.concat([search_results.drop('params', axis=1), params_df], axis=1)

            # 重新排列列顺序，将模型和目标放在前面
            cols = ['model', 'target'] + [col for col in search_results_expanded.columns if
                                          col not in ['model', 'target']]
            search_results_expanded = search_results_expanded[cols]

            search_results_expanded.to_csv(
                f'{hyperparameter_dir}/{target_name}_{model_name}_所有超参数组合性能.csv',
                index=False, encoding='utf-8-sig'
            )

    # 保存模型性能结果
    performance_dir = f"{results_dir}/模型性能分析"
    if not os.path.exists(performance_dir):
        os.makedirs(performance_dir)

    for target_name, performance in model_performance_results.items():
        # 保存基模型性能
        base_performance_data = []
        for model_name, perf in performance['base_models'].items():
            base_performance_data.append({
                '模型名称': model_name,
                '目标变量': target_name,
                '训练集R²': perf['train_r2'],
                '训练集RMSE': perf['train_rmse'],
                '训练集MAE': perf['train_mae'],
                '测试集R²': perf['test']['r2'],
                '测试集RMSE': perf['test']['rmse'],
                '测试集MAE': perf['test']['mae'],
                '交叉验证平均R²': perf['cv_mean'],
                '交叉验证标准差': perf['cv_std']
            })

        base_performance_df = pd.DataFrame(base_performance_data)
        base_performance_df.to_csv(
            f'{performance_dir}/{target_name}_基模型性能.csv',
            index=False, encoding='utf-8-sig'
        )

        # 保存元模型性能
        meta_performance_data = []
        for dataset, perf in performance['meta_model'].items():
            meta_performance_data.append({
                '模型类型': '元模型',
                '目标变量': target_name,
                '数据集': dataset,
                'R²': perf['r2'],
                'RMSE': perf['rmse'],
                'MAE': perf['mae']
            })

        meta_performance_df = pd.DataFrame(meta_performance_data)
        meta_performance_df.to_csv(
            f'{performance_dir}/{target_name}_元模型性能.csv',
            index=False, encoding='utf-8-sig'
        )

        # 保存基模型交叉验证分数
        cv_scores_data = []
        for model_name, scores in performance['base_models_cv'].items():
            for fold_idx, score in enumerate(scores['fold_scores']):
                cv_scores_data.append({
                    '模型名称': model_name,
                    '目标变量': target_name,
                    '折数': fold_idx + 1,
                    'R²分数': score
                })

        cv_scores_df = pd.DataFrame(cv_scores_data)
        cv_scores_df.to_csv(
            f'{performance_dir}/{target_name}_基模型交叉验证分数.csv',
            index=False, encoding='utf-8-sig'
        )

    # 保存总体性能汇总
    result_data = []
    for target_name in available_targets.keys():
        if target_name in results:
            result_data.append({
                '目标变量': target_name,
                '测试集样本数': len(results[target_name]['y_test']),
                '测试集R²': results[target_name]['r2'],
                '测试集RMSE': results[target_name]['rmse'],
                '测试集MAE': results[target_name]['mae'],
                '平均绝对误差': np.mean(np.abs(results[target_name]['y_test'] - results[target_name]['y_pred'])),
            })

    result_df = pd.DataFrame(result_data)
    print(f"\n{'=' * 60}")
    print("模型性能汇总")
    print(f"{'=' * 60}")
    print(result_df.round(4))
    result_df.to_csv(f'{results_dir}/模型性能汇总.csv', index=False, encoding='utf-8-sig')

    # 保存输入特征信息
    feature_info = pd.DataFrame({
        '输入特征类别': ['生化组成', '生化组成', '生化组成', '元素分析', '元素分析', '元素分析', '元素分析', '元素分析',
                         '灰分', '操作条件', '操作条件', '操作条件', '操作条件', '反应溶剂', '分离溶剂'],
        '特征名称': ['Pro (%)', 'Lip (%)', 'CH (%)', 'C (%)', 'H (%)', 'O (%)', 'N (%)', 'S (%)', 'Ash (%)',
                     'Time (min)', 'T (℃)', 'Size (L)', 'Ratio', 'RS', 'SS'],
        '是否在数据中': [col in data.columns for col in ['Pro (%)', 'Lip (%)', 'CH (%)', 'C (%)', 'H (%)', 'O (%)',
                                                         'N (%)', 'S (%)', 'Ash (%)', 'Time (min)', 'T (℃)',
                                                         'Size (L)', 'Ratio', 'RS', 'SS']]
    })
    feature_info.to_csv(f'{results_dir}/输入特征信息.csv', index=False, encoding='utf-8-sig')

    # 输出文件说明
    print(f"\n{'=' * 60}")
    print("生成的文件说明:")
    print(f"{'=' * 60}")
    print("\n输入特征类别:")
    print("- 生化组成: Pro (%) (蛋白质), Lip (%) (脂肪), CH (%) (碳水化合物)")
    print("- 元素分析: C (%), H (%), O (%), N (%), S (%)")
    print("- 灰分: Ash (%)")
    print("- 操作条件: Time (min) (停留时间), T (℃) (温度), Size (L) (反应器尺寸), Ratio (生物质与溶剂比)")
    print("- 反应溶剂: RS")
    print("- 分离溶剂: SS")

    print("\n目标变量（生物油性质）:")
    print("- Yield_oil: Yield_oil (wt%) (生物油产量)")
    print("- Oil_C: Oil_C (%) (生物油C含量)")
    print("- Oil_H: Oil_H (%) (生物油H含量)")
    print("- Oil_O: Oil_O (%) (生物油O含量)")
    print("- Oil_N: Oil_N (%) (生物油N含量)")
    print("- Oil_S: Oil_S (%) (生物油S含量)")
    print("- Oil_HHV: HHV (MJ/kg) (生物油热值)")

    for target_name in available_targets.keys():
        if target_name in results:
            print(f"\n{target_name}:")
            print(f"  - {target_name}_全部预测值.csv: 包含训练集和测试集的真实值和预测值")
            print(f"  - {target_name}_测试集完整数据.csv: 测试集所有特征数据和预测值")
            print(f"  - {target_name}_特征SHAP重要性排行.csv: 各输入特征的SHAP重要性排序")
            print(f"  - {target_name}_特征目标完整关系数据.csv: 输出特征随输入特征变化的完整数据")
            print(f"  - 超参数优化结果/: 包含各模型的超参数优化详细结果")
            print(f"    - {target_name}_[model]_最佳参数和性能.csv: 最佳参数和对应的R2/RMSE")
            print(f"    - {target_name}_[model]_所有超参数组合性能.csv: 所有超参数组合的R2和RMSE")
            print(f"  - 模型性能分析/: 包含各模型的训练集和测试集性能指标")

    print(f"\n输入特征信息已保存到: 输入特征信息.csv")
    print(f"\n生物油性质预测分析完成（使用5折交叉验证）！")
    print(f"所有结果已保存到 '{results_dir}' 目录")
    print(f"共分析了 {len(available_targets)} 个目标变量: {list(available_targets.keys())}")


# 运行主程序
if __name__ == "__main__":
    main()