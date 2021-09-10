from pathlib import Path
import pandas as pd
import shutil

def main():
    if not Path('data/train_images').exists():
        print('No train_images folder to sort!')
        return
    for i in range(5):
        Path(f'data/{i}').mkdir(exist_ok=True, parents=True)
    df = pd.read_csv('data/train.csv')
    for index, row in df.iterrows():
        source = Path(f'data/train_images/{row["image_id"]}')
        destination = Path(f'data/{row["label"]}')
        shutil.move(source, destination)

if __name__ == '__main__':
    main()