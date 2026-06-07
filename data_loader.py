"""
Unified dataset loading for SurrogateML experiments.
Each loader returns: (X_train, y_train, X_test, y_test, dataset_info_dict)
All features are numeric (float64). Categoricals are label-encoded.
"""

import numpy as np
import pandas as pd
import zipfile
from pathlib import Path
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "Datasets"


def _impute_nans(X_train, X_test):
    """Mean-impute any NaN values in train/test arrays."""
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        imp = SimpleImputer(strategy='mean')
        X_train = imp.fit_transform(X_train)
        X_test = imp.transform(X_test)
    return X_train, X_test


def load_heart(zip_path=None):
    """
    UCI Heart Disease: Cleveland (train) -> Switzerland (test).
    13 features, binary target (presence of heart disease).
    """
    if zip_path is None:
        zip_path = DATASET_DIR / "heart+disease.zip"
    zip_path = Path(zip_path)

    COLUMN_NAMES = [
        'age', 'sex', 'cp', 'trestbps', 'chol',
        'fbs', 'restecg', 'thalach', 'exang',
        'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]

    def _load_split(archive, filename):
        raw = archive.read(filename).decode('utf-8', errors='replace').replace('\x00', '')
        df = pd.read_csv(StringIO(raw), header=None, sep=",", na_values="?",
                         engine="python", on_bad_lines="warn")
        if df.shape[1] > len(COLUMN_NAMES):
            idx = [2, 3, 8, 9, 11, 15, 18, 31, 37, 39, 40, 43, 50, 57]
            df = df.iloc[:, idx]
        df.columns = COLUMN_NAMES
        df['target'] = (df['target'] > 0).astype(int)
        X = df.drop(columns=['target']).values.astype(float)
        y = df['target'].values
        return X, y

    with zipfile.ZipFile(zip_path, 'r') as archive:
        X_train, y_train = _load_split(archive, 'processed.cleveland.data')
        X_test, y_test = _load_split(archive, 'processed.switzerland.data')

    X_train, X_test = _impute_nans(X_train, X_test)

    info = {
        'name': 'heart',
        'n_features': X_train.shape[1],
        'n_classes': len(np.unique(y_train)),
        'train_source': 'Cleveland',
        'test_source': 'Switzerland',
    }
    return X_train, y_train, X_test, y_test, info


def load_mushroom(zip_path=None):
    """
    UCI Mushroom: Grass habitat (train) -> Woods habitat (test).
    22 features (label-encoded), binary target (edible/poisonous).
    """
    if zip_path is None:
        zip_path = DATASET_DIR / "mushroom.zip"
    zip_path = Path(zip_path)

    COLUMN_NAMES = [
        'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
        'stalk-surface-below-ring', 'stalk-color-above-ring',
        'stalk-color-below-ring', 'veil-type', 'veil-color',
        'ring-number', 'ring-type', 'spore-print-color',
        'population', 'habitat'
    ]

    HABITAT_MAP = {
        'g': 'grasses', 'l': 'leaves', 'm': 'meadows',
        'p': 'paths', 'u': 'urban', 'w': 'waste', 'd': 'woods'
    }

    with zipfile.ZipFile(zip_path, 'r') as archive:
        file_names = archive.namelist()
        data_file = next((f for f in file_names if "agaricus-lepiota.data" in f.lower()), None)
        if not data_file:
            raise FileNotFoundError("agaricus-lepiota.data not found in mushroom.zip")

        with archive.open(data_file) as f:
            df = pd.read_csv(f, header=None, na_values=["?", "NA", ""], engine="python")
            df.columns = COLUMN_NAMES

    # Binary target
    df['class'] = df['class'].map({'e': 0, 'p': 1})
    # Map habitat
    df['habitat'] = df['habitat'].map(HABITAT_MAP)

    # Label-encode all categorical features (except class and habitat used for splitting)
    for col in df.columns:
        if col not in ['class', 'habitat'] and df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Split by habitat: grasses -> train, woods -> test
    train_df = df[df['habitat'] == 'grasses'].copy()
    test_df = df[df['habitat'] == 'woods'].copy()

    # Drop habitat column (used for splitting, not a feature)
    feature_cols = [c for c in df.columns if c not in ['class', 'habitat']]
    X_train = train_df[feature_cols].values.astype(float)
    y_train = train_df['class'].values
    X_test = test_df[feature_cols].values.astype(float)
    y_test = test_df['class'].values

    X_train, X_test = _impute_nans(X_train, X_test)

    info = {
        'name': 'mushroom',
        'n_features': X_train.shape[1],
        'n_classes': 2,
        'train_source': 'grasses habitat',
        'test_source': 'woods habitat',
    }
    return X_train, y_train, X_test, y_test, info


def load_breast_cancer():
    """
    sklearn built-in Breast Cancer Wisconsin.
    30 features, binary target. 80/20 stratified split.
    """
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    info = {
        'name': 'breast_cancer',
        'n_features': X_train.shape[1],
        'n_classes': 2,
        'train_source': 'sklearn 80%',
        'test_source': 'sklearn 20%',
    }
    return X_train, y_train, X_test, y_test, info


def load_adult():
    """
    UCI Adult/Census Income. ~48k samples, 14 features, binary target (>50K).
    Downloads from UCI or uses local copy.
    """
    adult_dir = DATASET_DIR / "adult"
    train_path = adult_dir / "adult.data"
    test_path = adult_dir / "adult.test"

    COLUMNS = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Adult dataset not found at {adult_dir}. "
            "Run `python download_datasets.py` first."
        )

    df_train = pd.read_csv(train_path, header=None, names=COLUMNS,
                           na_values=' ?', skipinitialspace=True)
    df_test = pd.read_csv(test_path, header=None, names=COLUMNS,
                          na_values=' ?', skipinitialspace=True, skiprows=1)

    # Binary target
    df_train['income'] = df_train['income'].str.strip().str.rstrip('.')
    df_test['income'] = df_test['income'].str.strip().str.rstrip('.')
    df_train['income'] = (df_train['income'] == '>50K').astype(int)
    df_test['income'] = (df_test['income'] == '>50K').astype(int)

    # Label encode categoricals
    cat_cols = df_train.select_dtypes(include='object').columns
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([df_train[col], df_test[col]]).astype(str)
        le.fit(combined)
        df_train[col] = le.transform(df_train[col].astype(str))
        df_test[col] = le.transform(df_test[col].astype(str))

    feature_cols = [c for c in COLUMNS if c != 'income']
    X_train = df_train[feature_cols].values.astype(float)
    y_train = df_train['income'].values
    X_test = df_test[feature_cols].values.astype(float)
    y_test = df_test['income'].values

    X_train, X_test = _impute_nans(X_train, X_test)

    info = {
        'name': 'adult',
        'n_features': X_train.shape[1],
        'n_classes': 2,
        'train_source': 'UCI adult.data',
        'test_source': 'UCI adult.test',
    }
    return X_train, y_train, X_test, y_test, info


def load_credit():
    """
    UCI Default of Credit Card Clients. 30k samples, 23 features, binary target.
    """
    credit_dir = DATASET_DIR / "credit"

    # Try XLS first, then CSV
    xls_path = credit_dir / "default of credit card clients.xls"
    csv_path = credit_dir / "default%20of%20credit%20card%20clients.xls"

    if xls_path.exists():
        fpath = xls_path
    elif csv_path.exists():
        fpath = csv_path
    else:
        raise FileNotFoundError(
            f"Credit dataset not found at {credit_dir}. "
            "Run `python download_datasets.py` first."
        )

    try:
        df = pd.read_excel(fpath, header=1, engine='xlrd')
    except ImportError:
        try:
            df = pd.read_excel(fpath, header=1, engine='openpyxl')
        except Exception:
            df = pd.read_csv(fpath, header=1)
    except Exception:
        df = pd.read_csv(fpath, header=1)

    # Target column
    target_col = [c for c in df.columns if 'default' in c.lower()]
    if not target_col:
        target_col = [df.columns[-1]]
    target_col = target_col[0]

    # Drop ID column if present
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_test = _impute_nans(X_train, X_test)

    info = {
        'name': 'credit',
        'n_features': X_train.shape[1],
        'n_classes': 2,
        'train_source': 'UCI 80%',
        'test_source': 'UCI 20%',
    }
    return X_train, y_train, X_test, y_test, info


def load_letter():
    """
    UCI Letter Recognition. 20k samples, 16 features, 26-class.
    """
    letter_dir = DATASET_DIR / "letter"
    data_path = letter_dir / "letter-recognition.data"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Letter dataset not found at {letter_dir}. "
            "Run `python download_datasets.py` first."
        )

    df = pd.read_csv(data_path, header=None)
    # First column is the letter class
    y_raw = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.astype(float)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    info = {
        'name': 'letter',
        'n_features': X_train.shape[1],
        'n_classes': len(le.classes_),
        'train_source': 'UCI 80%',
        'test_source': 'UCI 20%',
    }
    return X_train, y_train, X_test, y_test, info


def load_california():
    """
    sklearn California Housing. 20k samples, 8 features.
    Regression target binarized at median for classification.
    """
    from sklearn.datasets import fetch_california_housing
    import ssl
    # Work around SSL cert issues on macOS
    try:
        data = fetch_california_housing()
    except Exception:
        ssl._create_default_https_context = ssl._create_unverified_context
        data = fetch_california_housing()
    X = data.data
    # Binarize at median
    y = (data.target >= np.median(data.target)).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    info = {
        'name': 'california',
        'n_features': X_train.shape[1],
        'n_classes': 2,
        'train_source': 'sklearn 80%',
        'test_source': 'sklearn 20%',
    }
    return X_train, y_train, X_test, y_test, info


def load_banknote():
    """
    UCI Banknote Authentication. 1372 samples, 4 features, binary.
    """
    banknote_dir = DATASET_DIR / "banknote"
    data_path = banknote_dir / "data_banknote_authentication.txt"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Banknote dataset not found at {banknote_dir}. "
            "Run `python download_datasets.py` first."
        )

    df = pd.read_csv(data_path, header=None)
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    info = {
        'name': 'banknote',
        'n_features': X_train.shape[1],
        'n_classes': 2,
        'train_source': 'UCI 80%',
        'test_source': 'UCI 20%',
    }
    return X_train, y_train, X_test, y_test, info


def load_adult_domainshift():
    """
    UCI Adult with domain shift: train on Male, test on Female.
    Significant covariate shift: male income rate 30.6%, female 10.9%.
    """
    adult_dir = DATASET_DIR / "adult"
    train_path = adult_dir / "adult.data"

    COLUMNS = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]

    df = pd.read_csv(train_path, header=None, names=COLUMNS,
                     na_values=' ?', skipinitialspace=True)
    df['income'] = (df['income'].str.strip().str.rstrip('.') == '>50K').astype(int)

    # Label encode categoricals
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    feature_cols = [c for c in COLUMNS if c not in ['income', 'sex']]
    # sex was already encoded: Male=1, Female=0 typically
    # Find the encoded value for Male/Female
    male_mask = df['sex'] == df['sex'].mode()[0]  # majority = Male
    if male_mask.sum() < len(df) / 2:
        male_mask = ~male_mask

    train_df = df[male_mask]
    test_df = df[~male_mask]

    X_train = train_df[feature_cols].values.astype(float)
    y_train = train_df['income'].values
    X_test = test_df[feature_cols].values.astype(float)
    y_test = test_df['income'].values

    X_train, X_test = _impute_nans(X_train, X_test)

    info = {
        'name': 'adult_ds',
        'n_features': X_train.shape[1],
        'n_classes': 2,
        'train_source': 'Male',
        'test_source': 'Female',
        'domain_shift': True,
    }
    return X_train, y_train, X_test, y_test, info


def load_california_domainshift():
    """
    California Housing with domain shift: train on South CA, test on North CA.
    Geographic covariate shift: different housing markets.
    """
    from sklearn.datasets import fetch_california_housing
    import ssl
    try:
        data = fetch_california_housing()
    except Exception:
        ssl._create_default_https_context = ssl._create_unverified_context
        data = fetch_california_housing()

    X = data.data
    y = (data.target >= np.median(data.target)).astype(int)

    # Latitude is feature index 6
    lat = X[:, 6]
    median_lat = np.median(lat)
    north = lat >= median_lat
    south = ~north

    # Train on South CA, test on North CA
    X_train = X[south]
    y_train = y[south]
    X_test = X[north]
    y_test = y[north]

    info = {
        'name': 'california_ds',
        'n_features': X_train.shape[1],
        'n_classes': 2,
        'train_source': 'South CA (lat < {:.1f})'.format(median_lat),
        'test_source': 'North CA (lat >= {:.1f})'.format(median_lat),
        'domain_shift': True,
    }
    return X_train, y_train, X_test, y_test, info


def load_credit_domainshift():
    """
    Credit Card Default with domain shift: train on younger (<34), test on older (>=34).
    Age-based covariate shift: different financial behaviors.
    """
    credit_dir = DATASET_DIR / "credit"
    xls_path = credit_dir / "default of credit card clients.xls"

    try:
        df = pd.read_excel(xls_path, header=1, engine='xlrd')
    except ImportError:
        try:
            df = pd.read_excel(xls_path, header=1, engine='openpyxl')
        except Exception:
            df = pd.read_csv(xls_path, header=1)

    target_col = [c for c in df.columns if 'default' in c.lower()]
    if not target_col:
        target_col = [df.columns[-1]]
    target_col = target_col[0]

    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    # Split by age
    age_col = 'AGE'
    median_age = df[age_col].median()
    young = df[df[age_col] < median_age]
    old = df[df[age_col] >= median_age]

    feature_cols = [c for c in df.columns if c != target_col]
    X_train = young[feature_cols].values.astype(float)
    y_train = young[target_col].values.astype(int)
    X_test = old[feature_cols].values.astype(float)
    y_test = old[target_col].values.astype(int)

    X_train, X_test = _impute_nans(X_train, X_test)

    info = {
        'name': 'credit_ds',
        'n_features': X_train.shape[1],
        'n_classes': 2,
        'train_source': 'Age < {:.0f}'.format(median_age),
        'test_source': 'Age >= {:.0f}'.format(median_age),
        'domain_shift': True,
    }
    return X_train, y_train, X_test, y_test, info


# Registry
DATASETS = {
    'heart': load_heart,
    'mushroom': load_mushroom,
    'breast_cancer': load_breast_cancer,
    'adult': load_adult,
    'credit': load_credit,
    'letter': load_letter,
    'california': load_california,
    'banknote': load_banknote,
    'adult_ds': load_adult_domainshift,
    'california_ds': load_california_domainshift,
    'credit_ds': load_credit_domainshift,
}


if __name__ == '__main__':
    for name, loader in DATASETS.items():
        try:
            X_tr, y_tr, X_te, y_te, info = loader()
            print(f"{name:15s}  train={X_tr.shape}  test={X_te.shape}  "
                  f"classes={info['n_classes']}  features={info['n_features']}")
        except FileNotFoundError as e:
            print(f"{name:15s}  MISSING: {e}")
        except Exception as e:
            print(f"{name:15s}  ERROR: {e}")
