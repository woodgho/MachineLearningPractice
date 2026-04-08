import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier, StackingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, auc,confusion_matrix
class TwoFeatureStumpClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        george = X.iloc[:, 26] > 0
        num650 = X.iloc[:, 27] > 0

        is_not_spam = george | num650
        return (~is_not_spam).astype(int).values  # 转换为 numpy 数组,不转化会导致StackingClassifier报错.在_concatenate_predictions方法中，它试图对自定义分类器的预测结果调用reshape方法。问题是TwoFeatureStumpClassifier的predict方法返回的是一个pandas Series，而不是NumPy数组，而pandas Series没有reshape方法。
    
    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
df = pd.read_csv("Project\\MachineLearning\\data\\spambase\\spambase.data", header=None)

# 特征数据形状: (4601, 57)
X = df.iloc[:, :-1] # print("特征数据形状:", X.shape)
Y = df.iloc[:, -1]  # print("标签数据形状:", Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,stratify=Y)  #采用分层抽样

#### 决策树
### 生成决策树(sklearn以cart算法实现)

clf_DecisionTree = DecisionTreeClassifier()
clf_DecisionTree = clf_DecisionTree.fit(X_train, Y_train)
path = clf_DecisionTree.cost_complexity_pruning_path(X_train, Y_train)

# ### 剪枝-Cost complexity pruning（原理未细看，剪枝策略）
# ccp_alphas, impurities = path.ccp_alphas, path.impurities

# ## 计算不同 alpha 值的交叉验证得分
# train_scores = []
# test_scores = []

# # 通过大体图像判断，缩小alpha范围
# alpha_range = np.arange(0.,0.02+0.001,0.001)
# impurities_range = []
# for alpha in alpha_range:
#     idx = np.argmin(np.abs(ccp_alphas - alpha))
#     impurities_range.append(impurities[idx])
    
# for alpha in alpha_range:
#     clf = DecisionTreeClassifier(ccp_alpha=alpha)
#     clf.fit(X_train, Y_train)
#     train_scores.append(clf.score(X_train, Y_train))
#     test_scores.append(clf.score(X_test, Y_test))

# # 绘制两张图
# fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# # 剪枝路径图
# ax[0].plot(alpha_range, impurities_range, marker='o', drawstyle="steps-post")
# ax[0].set_xlabel("Alpha")
# ax[0].set_ylabel("Total Impurity")
# ax[0].set_title("Impurity vs Alpha")

# # 准确率图
# ax[1].plot(alpha_range, train_scores, marker='o', label="Train", drawstyle="steps-post")
# ax[1].plot(alpha_range, test_scores, marker='o', label="Test", drawstyle="steps-post")
# ax[1].set_xlabel("Alpha")
# ax[1].set_ylabel("Accuracy")
# ax[1].set_title("Accuracy vs Alpha")
# ax[1].legend()

# plt.show()

# 最优alpha事实上就在（0.，0.005），差距不大，不妨取0.0025作为最优alpha
opt_alpha = 0.0025

clf_DecisionTree_opt = DecisionTreeClassifier(ccp_alpha=opt_alpha)
clf_DecisionTree_opt.fit(X_train, Y_train)

predictions_DecisionTree_opt = clf_DecisionTree_opt.predict(X_test)
print("最优alpha决策树准确率:", clf_DecisionTree_opt.score(X_test, Y_test))

#### 集成学习

### 弱分类器

## 决策树桩--max_depth=1的决策树
stump = DecisionTreeClassifier(max_depth=1)
stump.fit(X_train, Y_train)
predictions = stump.predict(X_test)
print("决策树桩准确率:", stump.score(X_test, Y_test))
# 绘制决策树
# plt.figure(figsize=(12, 8))
# plot_tree(stump, 
#           feature_names=X_train.columns,  # 特征名称
#           class_names=['Non-spam', 'spam'],  # 类别名称
#           filled=True,  
#           rounded=True,  
#           fontsize=10)  
# plt.title("Stump")
# plt.show()

## 朴素贝叶斯（假设服从高斯分布，GaussianNB经常被用于分类任务，如文本分类、垃圾邮件检测）
gnb=GaussianNB()
gnb.fit(X_train, Y_train)
predictions = gnb.predict(X_test)
print("朴素贝叶斯准确率:", gnb.score(X_test, Y_test))

## 小神经网络，设置早停，否则警告：
# 警告：ConvergenceWarning: Stochastic Optimizer: Maximum iterations (100) reached and the optimization hasn't converged yet.
mlp = MLPClassifier(hidden_layer_sizes=(5,), max_iter=100,learning_rate_init=0.01,  early_stopping=True,n_iter_no_change=10)
mlp.fit(X_train, Y_train)
predictions = mlp.predict(X_test)
print("小神经网络准确率:", mlp.score(X_test, Y_test))

## 简单判断： “乔治(27列)” 和 “650（28列）” 是非垃圾邮件的标志（spambase.DOCUMENTION）
# is_not_spam = (X.iloc[:, 26] > 0) | (X.iloc[:, 28] > 0)
# is_spam_prediction = (~is_not_spam).astype(int)
# accuracy = sum(is_spam_prediction == Y) / len(Y)
# print("简单判断准确率:", accuracy)
## 这两个应该就是一个决策树桩，我单独定义一个类来实现
custom_two_stump = TwoFeatureStumpClassifier()
custom_accuracy = custom_two_stump.score(X_test, Y_test)
print("简单判断决策树桩准确率:", custom_accuracy)

### Voting
voting_clf = VotingClassifier(
    estimators=[
        ('decision_tree_stump', stump),
        ('naive_bayes', gnb),
        ('neural_network', mlp),
        ('custom_stump', custom_two_stump)
    ],
    voting='hard' # 硬投票，取投票结果中出现次数最多的类别作为最终预测结果
)
voting_clf.fit(X_train, Y_train)
print("投票集成准确率:", voting_clf.score(X_test, Y_test))

### Bagging(Bootstrap Aggregating)
bagging_stump = BaggingClassifier(estimator=stump, n_estimators=50, max_samples=0.7, max_features=0.7, oob_score=True)
#oob检验需要尽可能多的采样和分类器，否则会有警告：没有足够的样本用于OOB估计
bagging_stump.fit(X_train, Y_train)
print("stump基分类Bagging的OOB准确率:", bagging_stump.oob_score_)  # 内置交叉验证结果
print("stump基分类Bagging的测试集准确率:", bagging_stump.score(X_test, Y_test))  # 独立测试集结果

bagging_gnb = BaggingClassifier(
    estimator=gnb, 
    n_estimators=10, 
    max_samples=0.7, 
    max_features=0.7,
)

bagging_mlp = BaggingClassifier(
    estimator=mlp, 
    n_estimators=10, 
    max_samples=0.7, 
    max_features=0.7,
)

# 训练各个Bagging模型
bagging_gnb.fit(X_train, Y_train)
bagging_mlp.fit(X_train, Y_train)

# 创建投票集成结合这些Bagging模型
hybrid_voting = VotingClassifier(
    estimators=[
        ('bagging_stump', bagging_stump),
        ('bagging_gnb', bagging_gnb),
        ('bagging_mlp', bagging_mlp)
    ],
    voting='soft'  # 软投票，根据分类器的预测概率取加权平均作为最终预测结果
)

hybrid_voting.fit(X_train, Y_train)
print("混合Bagging+投票集成准确率:", hybrid_voting.score(X_test, Y_test))

### Stacking
stacking_clf = StackingClassifier(
    estimators=[
        ('decision_tree_stump', stump),
        ('naive_bayes', gnb),
        ('neural_network', mlp),
        ('custom_stump', custom_two_stump)
    ],
    final_estimator=LogisticRegression(),
    cv=5,  # 5折交叉验证
)
stacking_clf.fit(X_train, Y_train)
print("Stacking集成准确率:", stacking_clf.score(X_test, Y_test))

### Random Forest
#具体实现中，随机特征数m选取的典型方案是m=⌊√N⌋或m=⌊logN⌋
#森林的大小T通常用500至数千不等
rf_clf = RandomForestClassifier(
    n_estimators=500,  # 树的数量
    max_depth=10,  # 树的最大深度，甚至可以就是一个树桩
    max_features='sqrt',  #每个树的最大特征数（sqrt(N)）or 'log2'表示log2(N)
    bootstrap=True, 
    oob_score=True,  
)
rf_clf.fit(X_train, Y_train)
print("随机森林OOB准确率:", rf_clf.oob_score_) 
print("随机森林测试集准确率:", rf_clf.score(X_test, Y_test)) 

### AdaBoost
ada_clf = AdaBoostClassifier(
    estimator=stump,  # 基分类器，这里使用决策树桩
    n_estimators=50,  
    learning_rate=1.0,  
    algorithm='SAMME',  # 算法类型，'SAMME'用于处理多分类问题，是课堂提到的，'SAMME.R'（默认值）：基于概率的SAMME算法，通常收敛更快，性能更好，但是SAMME.R算法（AdaBoost的默认算法）将在scikit-learn 1.6版本中因为稳定性问题被弃用（移除）
)
ada_clf.fit(X_train, Y_train)
print("AdaBoost准确率:", ada_clf.score(X_test, Y_test)) 
## 感觉可以类似混合Bagging+做成混合adaboost+，或者采用Stacking，这里就不实现了

## 模型评价指标
#混合Bagging+模型评估
hybrid_predictions = hybrid_voting.predict(X_test)
hybrid_proba = hybrid_voting.predict_proba(X_test)[:, 1]

# 计算各项指标
# 精确率
hybrid_precision = precision_score(Y_test, hybrid_predictions)
print(f"混合Bagging+投票集成精确率: {hybrid_precision:.4f}")

# 召回率
hybrid_recall = recall_score(Y_test, hybrid_predictions)
print(f"混合Bagging+投票集成召回率: {hybrid_recall:.4f}")

# F1分数
hybrid_f1 = f1_score(Y_test, hybrid_predictions)
print(f"混合Bagging+投票集成 F1分数: {hybrid_f1:.4f}") 

## AdaBoost模型评估
ada_predictions = ada_clf.predict(X_test)
ada_proba = ada_clf.predict_proba(X_test)[:, 1]

# 精确率
ada_precision = precision_score(Y_test, ada_predictions)
print(f"AdaBoost精确率: {ada_precision:.4f}")

# 召回率
ada_recall = recall_score(Y_test, ada_predictions)
print(f"AdaBoost召回率: {ada_recall:.4f}")

# F1分数
ada_f1 = f1_score(Y_test, ada_predictions)
print(f"AdaBoost F1分数: {ada_f1:.4f}") 
# 随机森林模型评估
rf_predictions = rf_clf.predict(X_test)
rf_proba = rf_clf.predict_proba(X_test)[:, 1]

# 计算各项指标
# 精确率
rf_precision = precision_score(Y_test, rf_predictions)
print(f"随机森林精确率: {rf_precision:.4f}")

# 召回率
rf_recall = recall_score(Y_test, rf_predictions)
print(f"随机森林召回率: {rf_recall:.4f}")

# F1分数
rf_f1 = f1_score(Y_test, rf_predictions)
print(f"随机森林 F1分数: {rf_f1:.4f}") 
## AdaBoost模型评估
ada_predictions = ada_clf.predict(X_test)
ada_proba = ada_clf.predict_proba(X_test)[:, 1]

# 精确率
ada_precision = precision_score(Y_test, ada_predictions)
print(f"AdaBoost精确率: {ada_precision:.4f}")

# 召回率
ada_recall = recall_score(Y_test, ada_predictions)
print(f"AdaBoost召回率: {ada_recall:.4f}")

# F1分数
ada_f1 = f1_score(Y_test, ada_predictions)
print(f"AdaBoost F1分数: {ada_f1:.4f}") 



## 
# # 混淆矩阵
# print("\n混淆矩阵:")
# cm = confusion_matrix(Y_test, ada_predictions)
# print(cm)

# # 绘制PR曲线
# precision, recall, _ = precision_recall_curve(Y_test, ada_proba)
# pr_auc = auc(recall, precision)

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(recall, precision, label=f'PR (AUC = {pr_auc:.4f})')
# plt.xlabel('Recall')    
# plt.ylabel('Precision')
# plt.title('AdaBoost PR')
# plt.legend()
# plt.grid(True)

# # 绘制ROC曲线
# fpr, tpr, _ = roc_curve(Y_test, ada_proba)
# roc_auc = auc(fpr, tpr)

# plt.subplot(1, 2, 2)
# plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})')
# plt.plot([0, 1], [0, 1], 'k--')  
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('AdaBoost ROC')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.savefig('d:/code/DataScience/task3/ada_evaluation_curves.png', dpi=300)
# plt.show()

# feature_importance = ada_clf.feature_importances_
# # 获取最重要的10个特征
# top_indices = np.argsort(feature_importance)[-10:][::-1]
# print("最重要的10个特征:")
# for i, idx in enumerate(top_indices):
#     print(f"{i+1}. 特征{idx}: 重要性 = {feature_importance[idx]:.4f}")

# # 绘制特征重要性图
# plt.figure(figsize=(10, 6))
# plt.bar(range(len(top_indices)), feature_importance[top_indices])
# plt.xticks(range(len(top_indices)), [f"{idx}" for idx in top_indices], rotation=45)
# plt.xlabel('Feature')
# plt.ylabel('Importance')
# plt.title('AdaBoost Model - Top 10 Important Features')
# plt.tight_layout()
# plt.savefig('d:/code/DataScience/task3/ada_feature_importance.png', dpi=300)
# plt.show()

# 绘制三个模型的PR曲线和ROC曲线对比图
plt.figure(figsize=(15, 6))

# PR曲线对比
plt.subplot(1, 2, 1)

# AdaBoost PR曲线
precision_ada, recall_ada, _ = precision_recall_curve(Y_test, ada_proba)
pr_auc_ada = auc(recall_ada, precision_ada)
plt.plot(recall_ada, precision_ada, label=f'AdaBoost (AUC = {pr_auc_ada:.4f})')

# 随机森林 PR曲线
precision_rf, recall_rf, _ = precision_recall_curve(Y_test, rf_proba)
pr_auc_rf = auc(recall_rf, precision_rf)
plt.plot(recall_rf, precision_rf, label=f'RandomForest (AUC = {pr_auc_rf:.4f})')

# 混合Bagging+投票集成 PR曲线
precision_hybrid, recall_hybrid, _ = precision_recall_curve(Y_test, hybrid_proba)
pr_auc_hybrid = auc(recall_hybrid, precision_hybrid)
plt.plot(recall_hybrid, precision_hybrid, label=f'Hybrid Bagging+Voting (AUC = {pr_auc_hybrid:.4f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve Comparison')
plt.legend()
plt.grid(True)

# ROC曲线对比
plt.subplot(1, 2, 2)

# AdaBoost ROC曲线
fpr_ada, tpr_ada, _ = roc_curve(Y_test, ada_proba)
roc_auc_ada = auc(fpr_ada, tpr_ada)
plt.plot(fpr_ada, tpr_ada, label=f'AdaBoost (AUC = {roc_auc_ada:.4f})')

# 随机森林 ROC曲线
fpr_rf, tpr_rf, _ = roc_curve(Y_test, rf_proba)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.plot(fpr_rf, tpr_rf, label=f'RandomForest (AUC = {roc_auc_rf:.4f})')

# 混合Bagging+投票集成 ROC曲线
fpr_hybrid, tpr_hybrid, _ = roc_curve(Y_test, hybrid_proba)
roc_auc_hybrid = auc(fpr_hybrid, tpr_hybrid)
plt.plot(fpr_hybrid, tpr_hybrid, label=f'Hybrid Bagging+Voting (AUC = {roc_auc_hybrid:.4f})')

plt.plot([0, 1], [0, 1], 'k--')  # 对角线
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('d:/code/DataScience/task3/models_comparison_curves.png', dpi=300)
plt.show()

# 模型性能对比表格
model_names = ['AdaBoost', 'RandomForest', 'Hybrid Bagging+Voting']
precisions = [ada_precision, rf_precision, hybrid_precision]
recalls = [ada_recall, rf_recall, hybrid_recall]
f1_scores = [ada_f1, rf_f1, hybrid_f1]
pr_aucs = [pr_auc_ada, pr_auc_rf, pr_auc_hybrid]
roc_aucs = [roc_auc_ada, roc_auc_rf, roc_auc_hybrid]

# 创建性能对比DataFrame
performance_df = pd.DataFrame({
    'Model': model_names,
    'Precision': precisions,
    'Recall': recalls,
    'F1-Score': f1_scores,
    'PR-AUC': pr_aucs,
    'ROC-AUC': roc_aucs
})

print("\n模型性能对比:")
print(performance_df.to_string(index=False))

# 保存性能对比表格为图片
plt.figure(figsize=(12, 6))
plt.axis('tight')
plt.axis('off')
table = plt.table(cellText=performance_df.values, colLabels=performance_df.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)
plt.title('Model Performance Comparison', fontsize=16)
plt.savefig('d:/code/DataScience/task3/models_performance_table.png', dpi=300, bbox_inches='tight')
plt.show()