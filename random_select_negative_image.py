import random
from pathlib import Path

DATA_DIR = './dataset/all_data'

def main():
    p = Path(DATA_DIR)
    all_files = [f for f in p.rglob('*.jpg') if '_-' in f.stem]

    total = len(all_files)
    sampled = random.sample(all_files, 1000)

    with open('./dataset/negative_images.txt', 'w') as f:
        for path in sampled:
            f.write(str(path) + '\n')


if __name__ == '__main__':
    main()