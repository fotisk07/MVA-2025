import torch
from torchvision import models
import torch.nn as nn

def precompute_features(
    model: models.ResNet, dataset: torch.utils.data.Dataset, device: torch.device
) -> torch.utils.data.Dataset:
    """
    Create a new dataset with the features precomputed by the model.

    If the model is $f \circ g$ where $f$ is the last layer and $g$ is
    the rest of the model, it is not necessary to recompute $g(x)$ at
    each epoch as $g$ is fixed. Hence you can precompute $g(x)$ and
    create a new dataset
    $\mathcal{X}_{\text{train}}' = \{(g(x_n),y_n)\}_{n\leq N_{\text{train}}}$

    Arguments:
    ----------
    model: models.ResNet
        The model used to precompute the features
    dataset: torch.utils.data.Dataset
        The dataset to precompute the features from
    device: torch.device
        The device to use for the computation

    Returns:
    --------
    torch.utils.data.Dataset
        The new dataset with the features precomputed
    """
    
    model.eval()
    model.to(device)
    
    classifier_name = 'fc'
    fc = getattr(model, classifier_name)
    setattr(model, classifier_name, nn.Identity())
    
    features = []
    labels = []
    
    with torch.no_grad():
        for x, y in dataset:
            feat = model(x.unsqueeze(0).to(device)).cpu()
            features.append(feat)
            labels.append(y)
    
    features = torch.cat(features, dim=0)
    labels = torch.tensor(labels)
    
    setattr(model, classifier_name, fc)
    
    return torch.utils.data.TensorDataset(features, labels)