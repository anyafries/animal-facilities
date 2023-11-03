"""Models for the project."""

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def get_resnet50():
  weights = ResNet50_Weights.DEFAULT
  model = resnet50(weights=weights)

  for param in model.parameters():
    param.requires_grad = False

  # model.fc = nn.Linear(2048, 1)
  model.fc = nn.Sequential(nn.Linear(2048, 256),
                           nn.ReLU(),
                           nn.Linear(256, 1))
  # model.fc = nn.Sequential(
  #   nn.Linear(2048, 512),
  #   nn.ReLU(),
  #   nn.Linear(512, 256),
  #   nn.ReLU(),
  #   nn.Linear(256, 1)
  # )

  # Unfreeze the last few layers of the model
  for param in model.layer4.parameters():
      param.requires_grad = True

  return model