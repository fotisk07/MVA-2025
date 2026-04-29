import pickle
import random
from typing import Any

import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Batch
from torchvision import transforms

from clip_reproduction.models.text import ByteTokenizer

PROMPT_TEMPLATES = {
    "cifar10": [
        "a photo of a {label}",
        "a blurry photo of a {label}",
        "a close-up photo of a {label}",
        "a low resolution photo of a {label}",
    ],
    "cifar100": [
        "a photo of a {label}",
        "a blurry photo of a {label}",
        "a close-up photo of a {label}",
        "a low resolution photo of a {label}",
    ],
    "fashionmnist": [
        "a photo of a {label}",
        "a blurry photo of a {label}",
        "a close-up photo of a {label}",
        "a low resolution photo of a {label}",
    ],
    "kmnist": [
        "a photo of a {label}",
        "a blurry photo of a {label}",
        "a close-up photo of a {label}",
        "a low resolution photo of a {label}",
    ],
    "mnist": [
        "a photo of the digit {label}",
        "a handwritten digit {label}",
        "an image of the number {label}",
    ],
    "eurosat": [
        "a satellite photo of {label}",
    ],
    "oxfordiiitpet": [
        "a photo of a {label}",
    ],
    "food101": [
        "a photo of {label}",
    ],
}

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD = (0.2675, 0.2565, 0.2761)
MNIST_MEAN = (0.1307, 0.1307, 0.1307)
MNIST_STD = (0.3081, 0.3081, 0.3081)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DATASET_SPECS = {
    "cifar10": {
        "class": torchvision.datasets.CIFAR10,
        "split_key": "train",
        "train_split": True,
        "test_split": False,
        "mean": CIFAR10_MEAN,
        "std": CIFAR10_STD,
        "hflip": True,
        "to_rgb": False,
    },
    "cifar100": {
        "class": torchvision.datasets.CIFAR100,
        "split_key": "train",
        "train_split": True,
        "test_split": False,
        "mean": CIFAR_MEAN,
        "std": CIFAR_STD,
        "hflip": True,
        "to_rgb": False,
    },
    "mnist": {
        "class": torchvision.datasets.MNIST,
        "split_key": "train",
        "train_split": True,
        "test_split": False,
        "mean": MNIST_MEAN,
        "std": MNIST_STD,
        "hflip": False,
        "to_rgb": True,
    },
    "fashionmnist": {
        "class": torchvision.datasets.FashionMNIST,
        "split_key": "train",
        "train_split": True,
        "test_split": False,
        "mean": MNIST_MEAN,
        "std": MNIST_STD,
        "hflip": False,
        "to_rgb": True,
    },
    "kmnist": {
        "class": torchvision.datasets.KMNIST,
        "split_key": "train",
        "train_split": True,
        "test_split": False,
        "mean": MNIST_MEAN,
        "std": MNIST_STD,
        "hflip": False,
        "to_rgb": True,
    },
    "eurosat": {
        "class": torchvision.datasets.EuroSAT,
        "split_key": "split",
        "train_split": "train",
        "test_split": "test",
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "hflip": True,
        "to_rgb": False,
    },
    "oxfordiiitpet": {
        "class": torchvision.datasets.OxfordIIITPet,
        "split_key": "split",
        "train_split": "trainval",
        "test_split": "test",
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "hflip": True,
        "to_rgb": False,
    },
    "food101": {
        "class": torchvision.datasets.Food101,
        "split_key": "split",
        "train_split": "train",
        "test_split": "test",
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "hflip": True,
        "to_rgb": False,
    },
}

# Backward-compatible alias.
DATASETS = {name: spec["class"] for name, spec in DATASET_SPECS.items()}


class ImageTextPairDataset(Dataset):
    def __init__(
        self,
        subset: Subset,
        class_names: list[str],
        tokenizer: ByteTokenizer,
        templates: list[str],
    ) -> None:
        self.subset = subset
        self.class_names = class_names
        self.tokenizer = tokenizer
        self.templates = templates

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.subset[idx]
        class_name = self.class_names[label]
        prompt = random.choice(self.templates).format(label=class_name)
        tokens = self.tokenizer.encode(prompt)
        return image, tokens


def _build_transform(name: str, image_size: int, is_train: bool):
    if name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {list(DATASET_SPECS.keys())}")

    spec = DATASET_SPECS[name]
    transforms_list = [transforms.Resize((image_size, image_size))]
    if spec["to_rgb"]:
        transforms_list.append(transforms.Grayscale(num_output_channels=3))
    if is_train and spec["hflip"]:
        transforms_list.append(transforms.RandomHorizontalFlip())
    transforms_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=spec["mean"], std=spec["std"]),
        ],
    )
    return transforms.Compose(transforms_list)


def _make_dataset(
    name: str,
    root: str,
    *,
    is_train: bool,
    transform,
    download: bool,
) -> Dataset:
    if name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {list(DATASET_SPECS.keys())}")

    spec = DATASET_SPECS[name]
    dataset_class = spec["class"]
    split_key = spec["split_key"]
    split_value = spec["train_split"] if is_train else spec["test_split"]

    if split_key == "train":
        kwargs = {"train": bool(split_value)}
        return dataset_class(root=root, transform=transform, download=download, **kwargs)

    kwargs = {split_key: split_value}
    try:
        return dataset_class(root=root, transform=transform, download=download, **kwargs)
    except TypeError as exc:
        # Older torchvision versions expose some datasets without a split argument.
        if split_key not in str(exc):
            raise
        full_ds = dataset_class(root=root, transform=transform, download=download)
        train_indices, test_indices = _split_indices(len(full_ds), val_ratio=0.2, seed=42)
        indices = train_indices if is_train else test_indices
        return Subset(full_ds, indices)


def _class_names(dataset: Dataset, name: str) -> list[str]:
    base = dataset
    while isinstance(base, Subset):
        base = base.dataset

    if hasattr(base, "classes"):
        return [str(x) for x in base.classes]
    if name in {"mnist", "fashionmnist", "kmnist"}:
        return [str(i) for i in range(10)]
    raise ValueError(f"Dataset '{name}' does not expose class names.")


def _split_indices(total_size: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(total_size))
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total_size, generator=generator).tolist()
    indices = [indices[i] for i in permutation]
    val_size = int(total_size * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def get_classification_datasets(
    name: str,
    root: str,
    image_size: int,
    val_ratio: float,
    seed: int,
    download: bool = True,
) -> tuple[Subset, Subset, int, list[str]]:
    if name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {list(DATASET_SPECS.keys())}")
    train_tf = _build_transform(name=name, image_size=image_size, is_train=True)
    eval_tf = _build_transform(name=name, image_size=image_size, is_train=False)

    train_base = _make_dataset(name=name, root=root, is_train=True, transform=train_tf, download=download)
    eval_base = _make_dataset(name=name, root=root, is_train=True, transform=eval_tf, download=download)

    class_names = _class_names(train_base, name=name)
    num_classes = len(class_names)

    train_indices, val_indices = _split_indices(len(train_base), val_ratio=val_ratio, seed=seed)
    train_subset = Subset(train_base, train_indices)
    val_subset = Subset(eval_base, val_indices)

    return train_subset, val_subset, num_classes, class_names


def get_classification_train_test_datasets(
    name: str,
    root: str,
    image_size: int,
    download: bool = True,
    train_transform=None,
    eval_transform=None,
) -> tuple[Dataset, Dataset, int, list[str]]:
    if name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {list(DATASET_SPECS.keys())}")

    train_tf = (
        train_transform
        if train_transform is not None
        else _build_transform(name=name, image_size=image_size, is_train=True)
    )
    eval_tf = (
        eval_transform
        if eval_transform is not None
        else _build_transform(name=name, image_size=image_size, is_train=False)
    )

    train_ds = _make_dataset(name=name, root=root, is_train=True, transform=train_tf, download=download)
    test_ds = _make_dataset(name=name, root=root, is_train=False, transform=eval_tf, download=download)

    class_names = _class_names(train_ds, name=name)
    num_classes = len(class_names)

    return train_ds, test_ds, num_classes, class_names


def get_clip_datasets(
    name: str,
    root: str,
    image_size: int,
    val_ratio: float,
    seed: int,
    tokenizer: ByteTokenizer,
    download: bool = True,
) -> tuple[ImageTextPairDataset, ImageTextPairDataset, int, list[str]]:
    train_subset, val_subset, num_classes, class_names = get_classification_datasets(
        name=name,
        root=root,
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed,
        download=download,
    )

    train_dataset = ImageTextPairDataset(
        subset=train_subset,
        class_names=class_names,
        tokenizer=tokenizer,
        templates=PROMPT_TEMPLATES[name],
    )
    val_dataset = ImageTextPairDataset(
        subset=val_subset,
        class_names=class_names,
        tokenizer=tokenizer,
        templates=PROMPT_TEMPLATES[name],
    )

    return train_dataset, val_dataset, num_classes, class_names


# =========================================================
# Molecule graph + text embedding datasets
# =========================================================
x_map: dict[str, list[Any]] = {
    "atomic_num": list(range(0, 119)),
    "chirality": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
        "CHI_TETRAHEDRAL",
        "CHI_ALLENE",
        "CHI_SQUAREPLANAR",
        "CHI_TRIGONALBIPYRAMIDAL",
        "CHI_OCTAHEDRAL",
    ],
    "degree": list(range(0, 11)),
    "formal_charge": list(range(-5, 7)),
    "num_hs": list(range(0, 9)),
    "num_radical_electrons": list(range(0, 5)),
    "hybridization": [
        "UNSPECIFIED",
        "S",
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
        "OTHER",
    ],
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
}

e_map: dict[str, list[Any]] = {
    "bond_type": [
        "UNSPECIFIED",
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "QUADRUPLE",
        "QUINTUPLE",
        "HEXTUPLE",
        "ONEANDAHALF",
        "TWOANDAHALF",
        "THREEANDAHALF",
        "FOURANDAHALF",
        "FIVEANDAHALF",
        "AROMATIC",
        "IONIC",
        "HYDROGEN",
        "THREECENTER",
        "DATIVEONE",
        "DATIVE",
        "DATIVEL",
        "DATIVER",
        "OTHER",
        "ZERO",
    ],
    "stereo": [
        "STEREONONE",
        "STEREOANY",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
    ],
    "is_conjugated": [False, True],
}


def load_id2emb(csv_path: str) -> dict[str, torch.Tensor]:
    df = pd.read_csv(csv_path)
    id2emb: dict[str, torch.Tensor] = {}
    for _, row in df.iterrows():
        id_ = str(row["ID"])
        emb_vals = [float(x) for x in str(row["embedding"]).split(",")]
        id2emb[id_] = torch.tensor(emb_vals, dtype=torch.float32)
    return id2emb


def load_descriptions_from_graphs(graph_path: str) -> dict[str, str]:
    with open(graph_path, "rb") as f:
        graphs = pickle.load(f)
    return {graph.id: graph.description for graph in graphs}


class PreprocessedGraphDataset(Dataset):
    def __init__(self, graph_path: str, emb_dict=None):
        with open(graph_path, "rb") as f:
            self.graphs = pickle.load(f)
        self.emb_dict = emb_dict

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        if self.emb_dict is not None:
            return graph, self.emb_dict[graph.id]
        return graph


def collate_graph_text_batch(batch):
    if isinstance(batch[0], tuple):
        graphs, text_embs = zip(*batch)
        batch_graph = Batch.from_data_list(list(graphs))
        return batch_graph, torch.stack(text_embs, dim=0)
    return Batch.from_data_list(batch)


# Backward-compatible alias for previous code paths.
collate_fn = collate_graph_text_batch
