"""
Download all UCI datasets needed for SurrogateML experiments.
Datasets already bundled as ZIPs (heart, mushroom) are skipped.
sklearn built-in datasets (breast_cancer, california) need no download.
"""
import urllib.request
import os
import ssl

# Disable SSL verification for UCI downloads (their cert is sometimes flaky)
ssl._create_default_https_context = ssl._create_unverified_context

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "Datasets")

DOWNLOADS = {
    'adult': {
        'urls': [
            'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
            'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
        ],
        'dest': os.path.join(DATASETS_DIR, 'adult'),
    },
    'credit': {
        'urls': [
            'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls',
        ],
        'dest': os.path.join(DATASETS_DIR, 'credit'),
    },
    'letter': {
        'urls': [
            'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data',
        ],
        'dest': os.path.join(DATASETS_DIR, 'letter'),
    },
    'banknote': {
        'urls': [
            'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt',
        ],
        'dest': os.path.join(DATASETS_DIR, 'banknote'),
    },
}


def download_all():
    for name, info in DOWNLOADS.items():
        os.makedirs(info['dest'], exist_ok=True)
        for url in info['urls']:
            fname = url.split('/')[-1]
            # URL-decode the filename
            fname = urllib.parse.unquote(fname)
            dest_path = os.path.join(info['dest'], fname)
            if not os.path.exists(dest_path):
                print(f"Downloading {name}: {fname}...")
                try:
                    urllib.request.urlretrieve(url, dest_path)
                    print(f"  -> Saved to {dest_path}")
                except Exception as e:
                    print(f"  FAILED: {e}")
                    print(f"  Try manual download from: {url}")
            else:
                print(f"Already exists: {dest_path}")


def download_via_openml():
    """Fallback: download via openml if UCI URLs fail."""
    try:
        import openml
    except ImportError:
        print("openml not installed. Run: pip install openml")
        return

    datasets_map = {
        'adult': 1590,
        'credit': 42477,
        'letter': 6,
    }

    for name, dataset_id in datasets_map.items():
        dest = os.path.join(DATASETS_DIR, name)
        if os.path.exists(dest) and os.listdir(dest):
            print(f"Already exists: {dest}")
            continue
        print(f"Downloading {name} via OpenML (ID={dataset_id})...")
        try:
            dataset = openml.datasets.get_dataset(dataset_id)
            X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
            os.makedirs(dest, exist_ok=True)
            import pandas as pd
            df = pd.concat([X, y], axis=1)
            csv_path = os.path.join(dest, f"{name}_openml.csv")
            df.to_csv(csv_path, index=False)
            print(f"  -> Saved to {csv_path}")
        except Exception as e:
            print(f"  FAILED: {e}")


if __name__ == '__main__':
    import urllib.parse
    print("=== Downloading datasets from UCI ===")
    download_all()
    print("\n=== Checking which datasets are available ===")
    from data_loader import DATASETS
    for name, loader in DATASETS.items():
        try:
            X_tr, y_tr, X_te, y_te, info = loader()
            print(f"  OK: {name:15s}  train={X_tr.shape}  test={X_te.shape}")
        except FileNotFoundError as e:
            print(f"  MISSING: {name:15s}  {e}")
        except Exception as e:
            print(f"  ERROR: {name:15s}  {e}")
