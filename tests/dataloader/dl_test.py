from torchssl.dataset.ssldataloader import SSLDataloader


path_dir = "tests/test_data/unlabeled_images"

dataloader = SSLDataloader(
    data_dir=path_dir,
    augmentation=None,
    batch_size=32,
    num_workers=3
)



for i in dataloader:
    print(i)