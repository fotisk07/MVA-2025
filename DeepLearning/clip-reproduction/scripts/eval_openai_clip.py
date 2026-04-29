import hydra
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPTokenizer

from clip_reproduction import utils
from clip_reproduction.datasets import get_classification_datasets


@hydra.main(version_base=None, config_path="../conf", config_name="train_conf")
def main(cfg) -> None:
    utils.seed_everything(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device", device)
    device = torch.device(device)

    train_ds, val_ds, num_classes, class_names = get_classification_datasets(
        name=cfg.dataset,
        root=cfg.data_root,
        image_size=cfg.image_size,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    model.eval()

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

    texts = [f"A photo of the digit {x}" if str(x).isdigit() 
             else f"A photo of a {x}" for x in class_names]

    text_inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    all_labels = []
    all_preds = []

    for images, labels in tqdm(val_loader):
        image_features = model.get_image_features(pixel_values=images.to(device))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T

        preds = torch.argmax(similarity, dim=1)

        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())

    all_labels = torch.concatenate(all_labels)
    all_preds = torch.concatenate(all_preds)

    accuracy = (all_preds == all_labels).float().mean()
    print(f"Accuracy: {accuracy.item():.4f}")


if __name__ == "__main__":
    main()
