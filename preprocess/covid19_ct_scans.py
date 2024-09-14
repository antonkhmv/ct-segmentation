import os
from typing import cast

from data import AbstractLoader, NIfTILoader, download_kaggle_dataset
import pandas as pd
from tqdm.auto import tqdm
from concurrent import futures

from train_model import get_dataset, KaggleDataset, Config

# Remove first N components from all paths
TRIM_FIRST_N = 2


def save_splits(data_dir, data_dir_2d, loader, key, value, num):
    metadata = []
    path = os.path.join(*value.split("/")[-TRIM_FIRST_N:])
    path_3d, path_2d = os.path.join(data_dir, path), os.path.join(data_dir_2d, path)
    image = loader.load_3d_image(path_3d)
    os.makedirs(os.path.dirname(os.path.abspath(path_2d)), exist_ok=True)
    start = int(len(image) * 0.2)
    end = int(len(image) * 0.8)
    for slice_num, image in enumerate(image[start: end]):
        base, ext = os.path.splitext(path_2d)
        path_2d_image = loader.save_2d_image(f"{base}_slice={slice_num}.tif", image)
        metadata.append({"num": num, "slice": slice_num, "key": key, "path": path_2d_image})
    return metadata


def generate_2d_dataset(data_dir, data_dir_2d, loader: AbstractLoader):
    manifest_2d = os.path.join(data_dir_2d, "metadata.csv")

    if os.path.exists(manifest_2d):
        return manifest_2d

    manifest = os.path.join(data_dir, "metadata.csv")
    df = pd.read_csv(manifest)
    items = []
    executor = futures.ProcessPoolExecutor(max_workers=16)
    jobs = []
    for i, row in df.iterrows():
        for key, value in row.items():
            jobs.append(executor.submit(save_splits, data_dir, data_dir_2d, loader, key, value, i))
    metadata = []
    for future in tqdm(futures.as_completed(jobs), total=len(jobs)):
        metadata.extend(future.result())
    metadata = pd.DataFrame(metadata)
    for _, group in metadata.groupby(["num", "slice"]):
        items.append({keys['key']: keys['path'] for _, keys in group.iterrows()})

    pd.DataFrame(items).to_csv(manifest_2d, index=False)
    return manifest_2d


def init_kaggle(username, key):
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    import kaggle

    kaggle.api.authenticate()


def main():
    dataset = cast(KaggleDataset, get_dataset(
        {
            "type": "KaggleDataset",
            "params": dict(
                type="2d",
                name="covid19-ct-scans",
                kaggle_path="andrewmvd/covid19-ct-scans",
                loader_name="NIfTILoader",
                image_key="ct_scan",
                mask_key="infection_mask",
            )
        }
    ))
    loader = NIfTILoader()
    config = Config()
    path = os.path.join(config.data_dir, dataset.name)
    path_2d = os.path.join(config.data_dir_2d, dataset.name)
    download_kaggle_dataset(dataset.kaggle_path, path)
    generate_2d_dataset(path, path_2d, loader)


if __name__ == "__main__":
    main()

