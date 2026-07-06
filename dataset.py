import os
import random
import shutil
import tarfile
import urllib.request

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DATA_URLS = [
    "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz",
    "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz",
]
IMG_SIZE = 256
JITTER_SIZE = 286


def download_facades(root="."):
    if os.path.exists(os.path.join(root, "facades")):
        return
    archive = os.path.join(root, "facades.tar.gz")
    last_error = None
    for url in DATA_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r, open(archive, "wb") as f:
                shutil.copyfileobj(r, f)
            last_error = None
            break
        except Exception as e:
            last_error = e
    if last_error is not None:
        raise last_error
    with tarfile.open(archive) as tar:
        tar.extractall(root)


class FacadesDataset(Dataset):
    def __init__(self, root="facades", split="train"):
        folder = os.path.join(root, split)
        self.files = sorted(
            os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")
        )
        self.split = split
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        combined = Image.open(self.files[idx]).convert("RGB")
        w, h = combined.size
        target = combined.crop((0, 0, w // 2, h))
        source = combined.crop((w // 2, 0, w, h))
        if self.split == "train":
            source = source.resize((JITTER_SIZE, JITTER_SIZE), Image.BICUBIC)
            target = target.resize((JITTER_SIZE, JITTER_SIZE), Image.BICUBIC)
            x = random.randint(0, JITTER_SIZE - IMG_SIZE)
            y = random.randint(0, JITTER_SIZE - IMG_SIZE)
            source = source.crop((x, y, x + IMG_SIZE, y + IMG_SIZE))
            target = target.crop((x, y, x + IMG_SIZE, y + IMG_SIZE))
            if random.random() < 0.5:
                source = source.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            source = source.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
            target = target.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        source = self.normalize(self.to_tensor(source))
        target = self.normalize(self.to_tensor(target))
        return source, target
