import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from clip_reproduction.models.transformer import ResidualAttentionBlock


class CNNModel(nn.Module):
    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ResNet50Features(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        model = torchvision.models.resnet50(weights=weights)
        model.fc = nn.Identity()
        self.model = model

    def encode_image_penultimate(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_image_penultimate(x)


class ResNet50Finetuning(nn.Module):
    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class VisionTransformerCLIP(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        embed_dim: int = 512,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size ({image_size}) must be divisible by patch_size ({patch_size})")

        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.normalize = normalize

        grid = image_size // patch_size
        num_patches = grid * grid
        self.num_tokens = 1 + num_patches

        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)

        self.class_embedding = nn.Parameter(torch.randn(width) * (width**-0.5))
        self.positional_embedding = nn.Parameter(torch.randn(self.num_tokens, width) * (width**-0.5))

        self.ln_pre = nn.LayerNorm(width)
        self.transformer = nn.ModuleList(
            [
                ResidualAttentionBlock(width, heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(layers)
            ]
        )
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.randn(width, embed_dim) * (width**-0.5))

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.class_embedding, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Expected input size {(self.image_size, self.image_size)}, got {(height, width)}"
            )

        x = self.conv1(x)
        x = x.flatten(2).transpose(1, 2)

        cls = self.class_embedding.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, self.width)
        x = torch.cat([cls, x], dim=1)

        x = x + self.positional_embedding.unsqueeze(0)
        x = self.ln_pre(x)

        for block in self.transformer:
            x = block(x)

        x = self.ln_post(x[:, 0, :])
        x = x @ self.proj

        if self.normalize:
            x = F.normalize(x, dim=-1)

        return x

class ResNet50Embedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        backbone = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT
        )
        for param in backbone.parameters():
            param.requires_grad = False

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.projection = nn.Linear(in_features, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embeddings = self.projection(features)
        return embeddings
