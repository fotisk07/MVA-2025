from torch import nn

from clip_reproduction.models.clip import CLIPModel, build_clip_model, build_openai_clip_model
from clip_reproduction.models.vision import CNNModel, ResNet50Features, ResNet50Finetuning


def create_model(name: str, **kwargs) -> nn.Module:
    key = name.lower()

    if key == "clip":
        return build_clip_model(**kwargs)

    if key == "openai_clip":
        return build_openai_clip_model(
            model_name=kwargs.get("openai_model_name", "openai/clip-vit-base-patch16"),
        )

    num_classes = kwargs.get("num_classes", 100)
    pretrained = kwargs.get("pretrained", True)

    if key == "cnn":
        return CNNModel(num_classes=num_classes)

    if key == "resnet50":
        return ResNet50Features(pretrained=pretrained)

    if key == "resnet50_finetuning":
        return ResNet50Finetuning(num_classes=num_classes)

    available = [
        "clip",
        "openai_clip",
        "cnn",
        "resnet50",
        "resnet50_finetuning",
    ]
    raise ValueError(f"Unknown model '{name}'. Available models: {available}")


def is_clip_model(model: nn.Module) -> bool:
    return isinstance(model, CLIPModel)
