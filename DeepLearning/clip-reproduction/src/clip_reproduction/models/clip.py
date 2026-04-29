import torch
from torch import nn
from torchvision import transforms
from transformers import CLIPModel as HFCLIPModel

from clip_reproduction.models.text import TextTransformerCLIP
from clip_reproduction.models.vision import ResNet50Embedding, VisionTransformerCLIP


class CLIPModel(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        initial_logit_scale: float = 1 / 0.07,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(initial_logit_scale)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(images)

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(token_ids)

    def forward(self, images: torch.Tensor, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        image_features = self.encode_image(images)
        text_features = self.encode_text(token_ids)

        scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_image = scale * (image_features @ text_features.t())
        logits_per_text = logits_per_image.t()

        batch_size = images.size(0)
        targets = torch.arange(batch_size, device=images.device)
        loss_i = nn.functional.cross_entropy(logits_per_image, targets)
        loss_t = nn.functional.cross_entropy(logits_per_text, targets)
        loss = 0.5 * (loss_i + loss_t)

        return {
            "loss": loss,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
        }


class OpenAIClipModel(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16") -> None:
        super().__init__()

        self.model_name = model_name
        self.image_size = 224
        self.model = HFCLIPModel.from_pretrained(model_name)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.get_image_features(pixel_values=images)

    def encode_text(self, token_ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        return self.model.get_text_features(input_ids=token_ids, attention_mask=attention_mask)

    def image_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )


def build_clip_model(
    image_size: int = 224,
    context_length: int = 77,
    embed_dim: int = 512,
    vision_width: int = 512,
    vision_layers: int = 8,
    vision_heads: int = 8,
    vision_patch_size: int = 16,
    text_width: int = 512,
    text_layers: int = 8,
    text_heads: int = 8,
    dropout: float = 0.0,
    encoder: str = "vit",
) -> CLIPModel:

    if encoder == "vit":
        image_encoder = VisionTransformerCLIP(
            image_size=image_size,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            embed_dim=embed_dim,
            dropout=dropout,
        )
    elif encoder == "resnet50":
        image_encoder = ResNet50Embedding(embed_dim=embed_dim)

    text_encoder = TextTransformerCLIP(
        context_length=context_length,
        width=text_width,
        layers=text_layers,
        heads=text_heads,
        embed_dim=embed_dim,
        dropout=dropout,
    )
    return CLIPModel(image_encoder=image_encoder, text_encoder=text_encoder)


def build_openai_clip_model(model_name: str = "openai/clip-vit-base-patch16") -> OpenAIClipModel:
    return OpenAIClipModel(model_name=model_name)
