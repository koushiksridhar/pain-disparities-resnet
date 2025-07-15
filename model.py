import torch
import torch.nn as nn
import torchvision.models as models
from config import Config

class ResNet18Regressor(nn.Module):
    def __init__(self, pretrained=Config.PRETRAINED):
        super(ResNet18Regressor, self).__init__()
        # Load the pretrained ResNet-18 model
        self.resnet = models.resnet18(pretrained=pretrained)

        # Freeze all layers initially
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Unfreeze the last 12 convolutional layers
        # ResNet18 structure: conv1 (1) + layer1 (4) + layer2 (4) + layer3 (4) + layer4 (4) = 17 conv layers
        # Last 12 conv layers = layer2 (4) + layer3 (4) + layer4 (4)
        for param in self.resnet.layer2.parameters():
            param.requires_grad = True
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        # Replace the final fully connected layer
        # ResNet18's final layer has 512 features, so we'll create a new layer
        # that maps from 512 features to 1 output (for regression)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, Config.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.HIDDEN_SIZE, 1)
        )
        
        # Ensure the new FC layers are trainable (they are by default)
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x, mask=None):
        # If mask is provided, we can use it to handle missing images
        # For samples where mask=0, we could replace with zeros or another placeholder
        if mask is not None:
            # Create a mask where 1s become 1s and 0s become 0s, expanded to match x's dimensions
            mask = mask.view(-1, 1, 1, 1).expand_as(x)
            # Zero out images where mask is 0
            x = x * mask
            
        # Pass through the ResNet model
        return self.resnet(x)