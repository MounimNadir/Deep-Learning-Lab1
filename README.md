# Lab 1 - Deep Learning (PyTorch) : Regression & Multi class Classification

**Université Abdelmalek Essaadi - FST Tanger - Master MBD**
Supervised by : Pr. ELAACHAK Lotfi

## Objective
Get familiar with the PyTorch library by building DNN / MLP models for a regression task and a multi class classification task.

## Files
| file | content |
|------|---------|
| `Part1_Regression_NYSE.ipynb` | Part 1 : regression on the [NYSE dataset](https://www.kaggle.com/datasets/dgawlik/nyse) |
| `Part2_Classification_Maintenance.ipynb` | Part 2 : classification on the [Machine Predictive Maintenance dataset](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) |

The notebooks download the datasets automatically with `kagglehub`, so they can be run directly in Google Colab or Kaggle.

## Part 1 : Regression (NYSE)
- **EDA** : shape, missing values, distributions, correlation matrix, price evolution of some companies, top volumes.
- **Task** : predict the `close` price from `open`, `low`, `high`, `volume` (log of volume + standardization).
- **Model** : MLP (Linear + ReLU layers) with a configurable number of layers / neurons.
- **GridSearch** : with `GridSearchCV` + `skorch` on learning rate, optimizer (Adam / SGD), epochs, hidden size, number of layers.
- **Curves** : Loss / Epochs and R² (used as "accuracy") / Epochs for train and test.
- **Regularization** : Dropout, L2 (weight decay), L1, BatchNorm, and all together, compared to the base model.

## Part 2 : Multi class classification (Predictive Maintenance)
- **Preprocessing** : remove ids, remove incoherent rows (Target vs Failure Type), drop `Target` (leakage), encode `Type` and the label, standardization.
- **EDA** : class distribution (very unbalanced ~96% "No Failure"), histograms, correlation, boxplots per class.
- **Data augmentation** : SMOTE on the train set only to balance the classes (5 classes after cleaning, the 'Random Failures' rows were all incoherent).
- **Model** : MLP with 5 outputs (one per class) and CrossEntropyLoss.
- **GridSearch** : same hyper-parameters, scoring = macro F1.
- **Curves** : Loss / Epochs and Accuracy / Epochs for train and test.
- **Metrics** : accuracy, precision, sensitivity (recall), specificity, F1 (macro and weighted), confusion matrix, on train and test.
- **Regularization** : Dropout, L2, L1, BatchNorm, all together, compared to the base model.

## Synthesis (what I learned)
- How to build a neural network in PyTorch with `nn.Module`, write the training loop (forward, loss, backward, optimizer step) and evaluate it.
- How to use `skorch` to plug a PyTorch model into sklearn `GridSearchCV` to find good hyper-parameters (lr, optimizer, epochs, architecture).
- EDA is important before training : in Part 1 the features are almost perfectly correlated with the target, and in Part 2 the classes are very unbalanced.
- With unbalanced data the accuracy is misleading, it is better to look at the macro F1, recall per class and the confusion matrix. SMOTE helps the model to learn the small classes, but it must be applied only on the train set.
- Data leakage : the `Target` column gives the answer, so it must be removed.
- Regularization (Dropout, L1, L2, BatchNorm) reduces overfitting (gap between train and test), but it doesn't help when the model is not overfitting, like in Part 1 where dropout even made the result worse. In Part 2, L1 gave the best macro F1 and dropout / L2 gave a better recall.
- Plotting Loss / Accuracy per epoch is a simple way to see if a model is underfitting or overfitting.
