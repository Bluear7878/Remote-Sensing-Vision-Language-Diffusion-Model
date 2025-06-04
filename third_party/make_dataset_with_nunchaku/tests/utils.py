import os

import datasets
import torch
import torchvision
from PIL import Image
from torch.utils import data
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm


def hash_str_to_int(s: str) -> int:
    """Hash a string to an integer."""
    modulus = 10**9 + 7  # Large prime modulus
    hash_int = 0
    for char in s:
        hash_int = (hash_int * 31 + ord(char)) % modulus
    return hash_int


def already_generate(save_dir: str, num_images) -> bool:
    if os.path.exists(save_dir):
        images = os.listdir(save_dir)
        images = [_ for _ in images if _.endswith(".png")]
        if len(images) == num_images:
            return True
    return False


class MultiImageDataset(data.Dataset):
    def __init__(self, gen_dirpath: str, ref_dirpath: str | datasets.Dataset):
        super(data.Dataset, self).__init__()
        self.gen_names = sorted(
            [name for name in os.listdir(gen_dirpath) if name.endswith(".png") or name.endswith(".jpg")]
        )
        self.ref_names = sorted(
            [name for name in os.listdir(ref_dirpath) if name.endswith(".png") or name.endswith(".jpg")]
        )
        self.gen_dirpath, self.ref_dirpath = gen_dirpath, ref_dirpath

        assert len(self.ref_names) == len(self.gen_names)
        self.transform = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.ref_names)

    def __getitem__(self, idx: int):
        name = self.ref_names[idx]
        assert name == self.gen_names[idx]
        ref_image = Image.open(os.path.join(self.ref_dirpath, name)).convert("RGB")
        gen_image = Image.open(os.path.join(self.gen_dirpath, name)).convert("RGB")
        gen_size = gen_image.size
        ref_size = ref_image.size
        if ref_size != gen_size:
            ref_image = ref_image.resize(gen_size, Image.Resampling.BICUBIC)
        gen_tensor = self.transform(gen_image)
        ref_tensor = self.transform(ref_image)
        return [gen_tensor, ref_tensor]


def compute_lpips(
    ref_dirpath: str, gen_dirpath: str, batch_size: int = 4, num_workers: int = 0, device: str | torch.device = "cuda"
) -> float:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    dataset = MultiImageDataset(gen_dirpath, ref_dirpath)
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    with torch.no_grad():
        desc = (os.path.basename(gen_dirpath)) + " LPIPS"
        for i, batch in enumerate(tqdm(dataloader, desc=desc)):
            batch = [tensor.to(device) for tensor in batch]
            metric.update(batch[0], batch[1])
    return metric.compute().item()
