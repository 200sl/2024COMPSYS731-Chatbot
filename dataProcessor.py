import os
import shutil
from torchvision import datasets
from torch.utils.data import random_split

raw_dataset = datasets.ImageFolder(root='dataset/raw', transform=None)

train_size = int(0.8 * len(raw_dataset))
test_size = len(raw_dataset) - train_size

train_dataset, test_dataset = random_split(raw_dataset, [train_size, test_size])


def save_dataset(processDataset, to_save_dir):
    for index in processDataset.indices:
        img_path, label = processDataset.dataset.imgs[index]
        label_dir = os.path.join(to_save_dir, processDataset.dataset.classes[label])
        os.makedirs(label_dir, exist_ok=True)

        img_filename = os.path.basename(img_path)
        target_path = os.path.join(label_dir, img_filename)

        # 复制图片到目标路径
        shutil.copy(img_path, target_path)


save_dataset(train_dataset, 'dataset/train_images')
save_dataset(test_dataset, 'dataset/test_images')
