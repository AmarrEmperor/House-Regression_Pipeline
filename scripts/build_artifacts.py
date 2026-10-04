"""Rebuild the housing notebook workspace without rerunning searches or plots."""
import argparse
import platform
import sys

from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from zlib import crc32
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import json
import cloudpickle
import sklearn

ROOT = Path(__file__).resolve().parents[1]

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))

def column_ratio(X):
    ### Calculate the ratio of the first column to the second column
    ### Input: X - a 2D numpy array
    ### Output: a 2D numpy array with the ratio of the first column to the second column
    return X[:, [0]] / X[:, [1]] 

def ratio_name(function_transformer, feature_names_in):
    # Return the names of the features after the transformation
    return["ratio"] # Feature names out must be a list of strings, so we return a list with a single string "ratio"

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler()
    )

log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler()
    )
cluster_sim = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_sim, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)  # one column remaining: housing_median_age


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing saved artifact set")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "saved_work",
                        help="Destination for generated artifacts")
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.exists() and any(destination.iterdir()) and not args.overwrite:
        parser.error(f"Artifacts already exist at {destination}; use --overwrite to replace them.")
    archive_path = ROOT / "data" / "housing.tgz"
    if not archive_path.is_file():
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading California housing data...", flush=True)
        urllib.request.urlretrieve("https://github.com/ageron/data/raw/main/housing.tgz", archive_path)
    with tarfile.open(archive_path) as archive:
        member = next(m for m in archive.getmembers() if m.name.endswith("housing.csv"))
        with archive.extractfile(member) as stream:
            housing = pd.read_csv(stream)
    target = "median_house_value"
    income_cat = pd.cut(housing["median_income"], bins=[0., 1.5, 3., 4.5, 6., np.inf],
                        labels=[1, 2, 3, 4, 5])
    strat_train_set, strat_test_set = train_test_split(
        housing, test_size=0.2, stratify=income_cat, random_state=42)
    housing_pre = strat_train_set.drop(columns=[target])
    housing_labels = strat_train_set[target].copy()
    target_scaler = StandardScaler()
    scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())
    models_to_fit = {
        "lin_reg": LinearRegression(),
        "tree_reg": DecisionTreeRegressor(random_state=42),
        "forest_reg": RandomForestRegressor(random_state=42),
        "svm_reg": SVR(kernel="rbf", C=10, epsilon=0.2, gamma=0.1),
    }
    from sklearn.base import clone
    fitted = {}
    for name, estimator in models_to_fit.items():
        print(f"Training {name}...", flush=True)
        pipeline = make_pipeline(clone(preprocessing), estimator)
        labels = housing_labels if name == "tree_reg" else scaled_labels.ravel()
        pipeline.fit(housing_pre, labels)
        fitted[name] = pipeline
    lin_reg, tree_reg, forest_reg, svm_reg = (fitted[name] for name in models_to_fit)
    full_pipeline = Pipeline([
        ("preprocessing", clone(preprocessing)),
        ("random_forest", RandomForestRegressor(random_state=42)),
    ])
    print("Training tuned random forest and saving workspace...", flush=True)
    from pathlib import Path
    import json
    import platform
    import cloudpickle
    import sklearn
    from sklearn.base import clone

    PROJECT_ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
                        if (p / "notebooks" / "notebook.ipynb").is_file())
    MODELS_PATH = destination
    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    # GridSearchCV refits automatically; explicitly retrain a separate final model.
    # Use the current search result when available; otherwise the recorded best params.
    best_forest_params = {"preprocessing__geo__n_clusters": 15,
                          "random_forest__max_features": 6}
    if "grid_search" in globals() and "random_forest__max_features" in grid_search.best_params_:
        best_forest_params = dict(grid_search.best_params_)
    final_forest_reg = clone(full_pipeline).set_params(**best_forest_params)
    final_forest_reg.fit(housing_pre, housing_labels)

    # Keep both the original 10-cluster features and the tuned model's 15-cluster features.
    saved_preprocessing = clone(preprocessing).fit(housing_pre)
    housing_prepared = saved_preprocessing.transform(housing_pre)
    housing_prepared_df = pd.DataFrame(housing_prepared,
        columns=saved_preprocessing.get_feature_names_out(), index=housing_pre.index)
    housing_test = strat_test_set.drop(columns=[target]).copy()
    housing_test_labels = strat_test_set[target].copy()
    housing_test_prepared = saved_preprocessing.transform(housing_test)
    final_preprocessing = final_forest_reg.named_steps["preprocessing"]
    housing_prepared_tuned = final_preprocessing.transform(housing_pre)
    housing_test_prepared_tuned = final_preprocessing.transform(housing_test)

    dataset = dict(target=target, housing_pre=housing_pre, housing_labels=housing_labels,
        scaled_labels=scaled_labels, housing_prepared=housing_prepared,
        housing_prepared_df=housing_prepared_df, strat_train_set=strat_train_set,
        strat_test_set=strat_test_set, housing_test=housing_test,
        housing_test_labels=housing_test_labels, housing_test_prepared=housing_test_prepared,
        housing_prepared_tuned=housing_prepared_tuned,
        housing_test_prepared_tuned=housing_test_prepared_tuned,
        tuned_feature_names=final_preprocessing.get_feature_names_out())
    models = dict(lin_reg=lin_reg, tree_reg=tree_reg, forest_reg=forest_reg,
                  svm_reg=svm_reg, final_forest_reg=final_forest_reg)
    # Cloudpickle includes notebook-defined transformers and functions for fresh kernels.
    def save_artifact(name, value):
        with (MODELS_PATH / name).open("wb") as stream:
            cloudpickle.dump(value, stream)

    # Original line: save_artifact("/dataset.pkl", dataset)
    save_artifact("dataset.pkl", dataset)
    save_artifact("preprocessing.pkl", saved_preprocessing)
    save_artifact("target_scaler.pkl", target_scaler)
    # Original incomplete call retained: save_artifact("")
    for name, estimator in models.items():
        save_artifact(f"{name}.pkl", estimator)
    manifest = dict(python=platform.python_version(), sklearn=sklearn.__version__,
        numpy=np.__version__, pandas=pd.__version__, cloudpickle=cloudpickle.__version__,
        best_forest_params=best_forest_params,
        prediction_targets={"lin_reg": "scaled", "tree_reg": "dollars",
            "forest_reg": "scaled", "svm_reg": "scaled", "final_forest_reg": "dollars"})
    (MODELS_PATH / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (MODELS_PATH / "requirements.txt").write_text(
        "\n".join(f"{name}=={version}" for name, version in {
            "scikit-learn": sklearn.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "cloudpickle": cloudpickle.__version__,
            # Original invalid package-version entry: "Linear_Regression":LinearRegression.__version__,
            # Original invalid package-version entry: "Decision_Tree_Regressor":DecisionTreeRegressor.__version__,
            # Original invalid package-version entry: "Random_Forest_Regressor":RandomForestRegressor.__version__,
            # Original invalid package-version entry: "Support_Vector_Machine":SVR.__version__,}.items()) + "\n",
        }.items()) + "\n",
        encoding="utf-8")
    print(f"Saved training/test data, preprocessing, target scaler and {len(models)} models to {MODELS_PATH}")


if __name__ == "__main__":
    main()
