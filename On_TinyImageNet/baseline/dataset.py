from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


def resolve_tiny_imagenet_root(root_dir):
    """Resolve either the dataset root or its common outer parent folder."""
    root = Path(root_dir).expanduser().resolve()
    candidates = [root, root / 'tiny-imagenet-200']
    for candidate in candidates:
        if (candidate / 'train').is_dir() and (candidate / 'val').is_dir():
            return candidate
    raise FileNotFoundError(
        'TinyImageNet root must contain train/ and val/. Checked: '
        + ', '.join(str(path) for path in candidates)
    )


class TinyImageNetDataset(Dataset):
    """Local tiny-imagenet-200 loader matching the layout used by VIB.py."""

    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = resolve_tiny_imagenet_root(root_dir)
        self.transform = transform
        self.train = bool(train)
        self.data = []
        self.labels = []

        class_names = sorted(
            path.name for path in (self.root_dir / 'train').iterdir()
            if path.is_dir()
        )
        self.class_to_idx = {
            class_name: index for index, class_name in enumerate(class_names)
        }
        if len(self.class_to_idx) != 200:
            raise ValueError(
                f'Expected 200 TinyImageNet classes, found '
                f'{len(self.class_to_idx)} in {self.root_dir / "train"}.'
            )

        if self.train:
            for class_name in class_names:
                image_dir = self.root_dir / 'train' / class_name / 'images'
                if not image_dir.is_dir():
                    raise FileNotFoundError(f'Missing image directory: {image_dir}')
                for image_path in sorted(image_dir.iterdir()):
                    if (
                        image_path.is_file()
                        and image_path.suffix.lower() in {'.jpeg', '.jpg', '.png'}
                    ):
                        self.data.append(image_path)
                        self.labels.append(self.class_to_idx[class_name])
        else:
            image_dir = self.root_dir / 'val' / 'images'
            annotation_path = self.root_dir / 'val' / 'val_annotations.txt'
            if not image_dir.is_dir() or not annotation_path.is_file():
                raise FileNotFoundError(
                    f'Expected {image_dir} and {annotation_path}.'
                )
            with annotation_path.open('r', encoding='utf-8') as handle:
                for line_number, line in enumerate(handle, start=1):
                    fields = line.rstrip('\n').split('\t')
                    if len(fields) < 2:
                        raise ValueError(
                            f'Malformed validation annotation at line '
                            f'{line_number}: {line!r}'
                        )
                    file_name, class_name = fields[:2]
                    if class_name not in self.class_to_idx:
                        raise ValueError(
                            f'Unknown validation class {class_name!r}.'
                        )
                    self.data.append(image_dir / file_name)
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(self.data[index]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]
