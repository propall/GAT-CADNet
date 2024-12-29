import os

import gdown


def main():
    dataset_dir = "../dataset"

    # download
    print(f'downloading train...')
    url = f"https://drive.google.com/uc?id=16McNNY_-Y2uVnq42ntZTdYKPWgOZxwp3"
    gdown.download(url, f"{dataset_dir}/train.zip")
    print(f'downloading val...')
    url = f"https://drive.google.com/uc?id=1xgLqcj91i13_3vhfsUYcRYh3PhFYB9LJ"
    gdown.download(url, f"{dataset_dir}/val.zip")
    print(f'downloading test...')
    url = f"https://drive.google.com/uc?id=1Hc4-ggsUMoB_5uqJdqYRn9K73QS8rOgG"
    gdown.download(url, f"{dataset_dir}/test.zip")

    # unzip
    zip_path = os.path.join(dataset_dir, "train.zip")
    unzip_dir = os.path.join(dataset_dir, "train")
    os.makedirs(unzip_dir, exist_ok=True)
    cmd = f"unzip {zip_path} -d {unzip_dir}"
    os.system(cmd)

    zip_path = os.path.join(dataset_dir, "val.zip")
    unzip_dir = os.path.join(dataset_dir, "val")
    os.makedirs(unzip_dir, exist_ok=True)
    cmd = f"unzip {zip_path} -d {unzip_dir}"
    os.system(cmd)

    zip_path = os.path.join(dataset_dir, "test.zip")
    unzip_dir = os.path.join(dataset_dir, "test")
    os.makedirs(unzip_dir, exist_ok=True)
    cmd = f"unzip {zip_path} -d {unzip_dir}"
    os.system(cmd)

if __name__ == "__main__":
    main()



