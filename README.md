# California Housing Price Prediction

This project explores a complete machine learning workflow for predicting the
median house value of California districts. It uses demographic, geographic,
and housing attributes to compare regression models and investigate how feature
engineering and hyperparameter tuning affect prediction accuracy.

The work is organized around two notebooks: an original notebook containing the
analysis and training process, and a continuation notebook that restores saved
data and fitted models so further experiments can begin without repeating that
process.

## Dataset and objective

The housing dataset is retrieved from the
[Hands-On Machine Learning data repository](https://github.com/ageron/data).
It contains 20,640 observations, with attributes including longitude, latitude,
housing age, room and bedroom counts, population, households, median income,
and proximity to the ocean. The prediction target is `median_house_value`,
expressed in dollars. Each observation represents a district rather than an
individual house.

The data is divided into 80% training and 20% test sets, stratified by income
category with a fixed random seed of 42. This keeps the income distribution
similar across the two sets. Preprocessing is fitted on the training data and
then applied to the test data.

## Modeling workflow

The original notebook covers the following stages:

1. **Explore the data:** inspect missing values, attribute distributions,
   geographic patterns, and correlations with house values.
2. **Prepare features:** impute missing numeric values, encode ocean proximity,
   and standardize numeric attributes. Add ratios such as rooms per household,
   logarithmic transformations, and geographic cluster similarities.
3. **Train and compare models:** fit linear regression, a decision tree, a
   random forest, and an RBF support vector regressor (SVR). Use cross-validation
   and root mean squared error (RMSE) to assess performance.
4. **Tune hyperparameters:** use grid search to explore SVR settings and random
   forest settings, including the number of geographic clusters.
5. **Save the workspace:** persist prepared features, targets, preprocessing,
   the target scaler, and fitted model pipelines for later use.

There are four baseline models and an additional tuned random forest. The
recorded best forest configuration uses 15 geographic clusters and
`max_features=6`. The baseline feature matrix has 24 columns; the tuned forest's
matrix has 29 because it uses five additional geographic clusters.

## Environment setup

The commands below assume Windows PowerShell and that your terminal is open in
the project root. Use Python 3.12 to match the current saved artifact manifest.

Create an isolated environment and install the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Register the environment as a notebook kernel and launch JupyterLab:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name house --display-name "House (.venv)"
.\.venv\Scripts\python.exe -m jupyterlab
```

Select **House (.venv)** when opening either notebook. The requirements file
installs the packages; the notebook's import cells make them available to your
session. You can also open the notebooks in VS Code and select the same
environment. On other operating systems, use the equivalent virtual environment
commands.

## Executing the project

### Continue from the saved workspace

Use this path when `models/saved_work/` already contains your saved artifacts.

1. Open [`notebooks/continue_work.ipynb`](notebooks/continue_work.ipynb).
2. Run the first code cell to import the packages.
3. Run the second code cell to load the datasets, targets, preprocessing,
   target scaler, and all five fitted models.
4. Run the third code cell to define `predict_dollars()` and evaluate the tuned
   random forest on the saved test set.
5. Continue with your experiments in the later cells.

The loading cells do not train models or repeat preprocessing. Later cells
contain additional search experiments, so **Run All** will execute those too.

The restored variables include:

| Variables | Contents |
| --- | --- |
| `housing_pre`, `housing_test` | Raw training and test features |
| `X_train`, `X_test` | Prepared features with 24 columns |
| `X_train_tuned`, `X_test_tuned` | Prepared features with 29 columns |
| `y_train`, `y_test` | House values in dollars |
| `scaled_labels` | Standardized training targets |
| `models` | The four baseline pipelines and tuned forest pipeline |

Saved models include their own preprocessing, so pass **raw feature DataFrames**
to them. For example:

```python
predictions = predict_dollars("final_forest_reg", housing_test)
rmse = root_mean_squared_error(y_test, predictions)
print(f"Test RMSE: ${rmse:,.2f}")
```

Linear regression, the baseline random forest, and SVR were fitted on scaled
targets. The helper converts their predictions back to dollars. The decision
tree and tuned forest already predict in dollars. Use the prepared matrices
when fitting new estimators directly, without another preprocessing pipeline.

### Run the original analysis from the beginning

Open [`notebooks/notebook.ipynb`](notebooks/notebook.ipynb) and execute its cells
in order. Set the notebook's working directory to `notebooks/`, since its
original data and image paths are relative to that directory.

This runs the exploration, visualizations, preprocessing demonstrations, model
training, cross-validation, and grid searches. The searches take longer than
loading an existing workspace. Once training is complete, run the saving cell
near the end to refresh `models/saved_work/` with the current fitted models and
datasets. You can then switch to the continuation notebook.

### Create artifacts when none are available

Datasets and fitted model binaries are excluded from Git, so a fresh clone does
not contain the saved workspace. You can run the original notebook as described
above, or use this optional shortcut from the project root:

```powershell
.\.venv\Scripts\python.exe scripts/build_artifacts.py
```

The script downloads the housing archive if needed, reproduces the training/test
split, fits preprocessing and all five models, and saves the files expected by
the continuation notebook. It uses the recorded winning hyperparameters and
skips plots, cross-validation, and grid searches. Training can take several
minutes.

The script leaves existing artifacts intact unless you explicitly request a
replacement:

```powershell
.\.venv\Scripts\python.exe scripts/build_artifacts.py --overwrite
```

## Project structure

| Path | Purpose |
| --- | --- |
| `notebooks/notebook.ipynb` | Original exploration, preprocessing, training, and tuning |
| `notebooks/continue_work.ipynb` | Restore saved work and continue experimenting |
| `data/` | Housing archive and other local datasets |
| `models/saved_work/` | Prepared datasets, targets, scalers, fitted models, and version manifest |
| `images/` | Figures generated during exploration |
| `requirements.txt` | Dependencies for the workstation environment |
| `scripts/build_artifacts.py` | Optional shortcut for recreating the saved workspace |
| `scripts/check_workspace.py` | Notebook structure and Python syntax validation used by GitHub checks |

Downloaded datasets, saved model binaries, and generated figures stay local.
Other existing files in `data/` are not inputs to the housing rebuild script.

## Saved model compatibility

`models/saved_work/manifest.json` records the environment versions and whether
each model predicts scaled targets or dollar values. Use those package versions
when restoring an existing artifact set. If you upgrade the modeling environment,
retrain and save a new compatible set of artifacts. Only load pickle files from
a trusted source.
