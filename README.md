# Lab 1 - Deep Learning (PyTorch) : Regression & Multi class Classification

**Université Abdelmalek Essaadi - FST Tanger - Master MBD**

Realized by : **Mounim Nadir**

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

## Results

### Part 1 : Regression (NYSE)
Best hyper-parameters (GridSearch, CV R² = 0.9998) : Adam, lr = 0.001, 40 epochs, 2 hidden layers of 32 neurons.

Test set (prices in $) :

| model | MSE | MAE | R² |
|-------|-----|-----|----|
| **base** | **0.66** | **0.47** | **0.9999** |
| dropout (0.2) | 106.34 | 3.14 | 0.9799 |
| L2 (1e-3) | 1.30 | 0.58 | 0.9998 |
| L1 (1e-5) | 0.96 | 0.55 | 0.9998 |
| batchnorm | 18.62 | 3.19 | 0.9965 |
| all together | 69.11 | 6.93 | 0.9869 |

Train and test curves are almost the same, so the base model is not overfitting and regularization does not improve it.

### Part 2 : Multi class classification (Predictive Maintenance)
Best hyper-parameters (GridSearch, CV macro F1 = 0.989) : Adam, lr = 0.01, 40 epochs, 2 hidden layers of 64 neurons.

Base model on train and test :

| data | accuracy | precision (macro) | recall (macro) | F1 (macro) | F1 (weighted) |
|------|----------|-------------------|----------------|------------|---------------|
| train (balanced) | 0.994 | 0.994 | 0.994 | 0.994 | 0.994 |
| train (original) | 0.981 | 0.732 | 0.978 | 0.811 | 0.984 |
| test | 0.962 | 0.574 | 0.783 | 0.645 | 0.969 |

Regularization comparison (test set) :

| model | accuracy | precision (macro) | recall (macro) | F1 (macro) |
|-------|----------|-------------------|----------------|------------|
| base | 0.962 | 0.574 | 0.783 | 0.645 |
| dropout (0.3) | 0.925 | 0.511 | 0.897 | 0.609 |
| L2 (1e-3) | 0.927 | 0.513 | 0.887 | 0.605 |
| **L1 (1e-5)** | **0.969** | **0.616** | 0.807 | **0.688** |
| batchnorm | 0.960 | 0.586 | 0.811 | 0.660 |
| all together | 0.871 | 0.445 | **0.951** | 0.540 |

The hardest class is **Tool Wear Failure** (only 9 samples in the test set, recall 0.44).

## Limitations
- **Part 1** : `open`, `high` and `low` of the same day are only known at the end of the day, and they are almost equal to `close`, so the task is easy (R² ≈ 1). The random split also mixes days from the same stock in train and test. A real forecasting task would predict the next day's close with a split by date.
- **Part 2** : the GridSearch cross-validation was done on the SMOTE data, so synthetic samples from one fold are close to samples of the other folds. This is why the CV macro F1 (0.989) is much higher than the test macro F1 (0.645). A better way is to put SMOTE inside an `imblearn` `Pipeline` so it is applied only on the training folds.
- The regularization models are compared on the test set, a separate validation set would be cleaner.

## Synthesis (what I learned)
- How to build a neural network in PyTorch with `nn.Module`, write the training loop (forward, loss, backward, optimizer step) and evaluate it.
- How to use `skorch` to plug a PyTorch model into sklearn `GridSearchCV` to find good hyper-parameters (lr, optimizer, epochs, architecture).
- EDA is important before training : in Part 1 the features are almost perfectly correlated with the target, and in Part 2 the classes are very unbalanced.
- With unbalanced data the accuracy is misleading, it is better to look at the macro F1, recall per class and the confusion matrix. SMOTE helps the model to learn the small classes, but it must be applied only on the train set.
- Data leakage : the `Target` column gives the answer, so it must be removed.
- Regularization (Dropout, L1, L2, BatchNorm) reduces overfitting (gap between train and test), but it doesn't help when the model is not overfitting, like in Part 1 where dropout even made the result worse. In Part 2, L1 gave the best macro F1 and dropout / L2 gave a better recall.
- Plotting Loss / Accuracy per epoch is a simple way to see if a model is underfitting or overfitting.
