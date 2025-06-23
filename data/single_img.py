import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as trans_fn


class SingleImageDataset(Dataset):
    def __init__(self, image_path, scale=1, resample=Image.BICUBIC, transform=None):
        self.image_path = image_path
        self.scale = scale
        self.resample = resample
        self.transform = transform if transform else transforms.ToTensor()
        self.image = Image.open(self.image_path).convert("RGB")
        self.resized_image = self.resize_and_convert(self.image)

    def resize_and_convert(self, img):
        orig_w, orig_h = img.size
        target_size = int(max(orig_w, orig_h) * self.scale)
        img = trans_fn.resize(img, target_size, self.resample)
        img = trans_fn.center_crop(img, target_size)
        return img

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img = self.transform(self.resized_image)
        return {'SR': img, 'Index': 0}

def single_image_dataloader(image_path, scale=1, resample=Image.BICUBIC):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    dataset = SingleImageDataset(
        image_path=image_path,
        scale=scale,
        resample=resample,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    return dataloader
