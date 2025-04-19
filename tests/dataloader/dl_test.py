from torchssl.dataset.ssldataloader import SSLDataloader
from torchssl.dataset.sslaug import SimclrAug

path_dir = "tests/test_data/unlabeled_images"


dataloader = SSLDataloader(
    data_dir=path_dir,
    augmentation=SimclrAug(img_size=224),
    batch_size=32,
    num_workers=3
)

for i in dataloader:
    print(i)