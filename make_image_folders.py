from pathlib import Path
import pandas as pd
import shutil

def main():
    for i in range(5):
        Path(f'data/{i}').mkdir(exist_ok=True, parents=True)
    df = pd.read_csv('data/train.csv')
    for index, row in df.iterrows():
        source = Path(f'data/train_images/{row["image_id"]}')
        destination = Path(f'data/{row["label"]}')
        shutil.move(source, destination)
    Path('data/train_images').rmdir()

if __name__ == '__main__':
    main()