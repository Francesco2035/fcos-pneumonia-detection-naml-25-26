
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class block(nn.Module):
    """
    Bottleneck residual block used in ResNet-50, ResNet-101 and ResNet-152.

    Architecture (main branch):

        Input
          │
       1×1 Conv
          │
      BatchNorm
          │
        ReLU
          │
       3×3 Conv
          │
      BatchNorm
          │
        ReLU
          │
       1×1 Conv
          │
      BatchNorm
          │
     + Identity Shortcut
          │
        ReLU
          │
        Output

    The residual function F(x) is computed by the three convolutions.
    The shortcut carries the original input x.
    The block output is:
            H(x) = F(x) + x
    """

    def __init__(self, in_channels, out_channels,
                 identity_downsample=None,
                 stride=1):

        super(block, self).__init__()

        # Expansion factor of the bottleneck architecture.
        #
        # The first two convolutions work on 'out_channels',
        # while the last 1×1 convolution expands the number
        # of feature maps by a factor of 4.
        #
        # Examples:
        # 64  -> 256
        # 128 -> 512
        # 256 -> 1024
        # 512 -> 2048
        self.expansion = 4

        # ----------------------------------------------------
        # First 1×1 convolution
        # ----------------------------------------------------
        #
        # Receives the input feature maps and projects them
        # into the bottleneck space.
        #
        # This layer reduces (or keeps) the number of channels
        # before the computationally expensive 3×3 convolution.
        #
        # Spatial resolution does NOT change
        # because stride = 1.
        #
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        # ----------------------------------------------------
        # Second 3×3 convolution
        # ----------------------------------------------------
        #
        # This is the main convolution of the bottleneck block.
        #
        # It extracts spatial features while working on the
        # reduced number of channels produced by conv1.
        #
        # If stride = 2 (first block of a new stage),
        # this convolution halves the spatial resolution:
        #
        # 56×56 -> 28×28
        # 28×28 -> 14×14
        # 14×14 -> 7×7
        #
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        # ----------------------------------------------------
        # Final 1×1 convolution
        # ----------------------------------------------------
        #
        # Restores the channel dimension after the bottleneck.
        #
        # Example:
        #
        # 64  -> 256
        # 128 -> 512
        # 256 -> 1024
        #
        # This produces the final residual function F(x).
        #
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.bn3 = nn.BatchNorm2d(
            out_channels * self.expansion
        )

        # Activation used after BatchNorm
        self.relu = nn.ReLU()

        # ----------------------------------------------------
        # Projection shortcut
        # ----------------------------------------------------
        #
        # Used only when the input tensor and the output tensor
        # have different shapes.
        #
        # This happens when:
        #
        # 1) the spatial resolution changes
        #    (stride = 2)
        #
        # or
        #
        # 2) the number of channels changes
        #
        # The projection is implemented with:
        #
        # 1×1 Conv + BatchNorm
        #
        # exactly as described in the ResNet paper.
        #
        self.identity_downsample = identity_downsample

    def forward(self, x):

        # Save the original input.
        # This tensor will travel through the shortcut branch.
        identity = x

        # ===========================
        # Residual branch (F(x))
        # ===========================

        # First bottleneck layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Main 3×3 convolution
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Channel expansion
        x = self.conv3(x)
        x = self.bn3(x)

        # ===========================
        # Shortcut branch
        # ===========================

        # If the tensor dimensions do not match,
        # transform the shortcut using a projection.
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        # ===========================
        # Residual connection
        # ===========================
        #
        # Implements the key equation of ResNet:
        #
        # H(x) = F(x) + x
        #
        # where:
        #
        # F(x) -> output of the three convolutions
        # x    -> shortcut connection
        #
        x += identity

        # Final non-linearity of the residual block
        x = self.relu(x)

        return x

class ResNet(nn.Module):
    """
    Implementation of the ResNet architecture.

    The parameter 'layers' specifies how many Bottleneck blocks
    are used in each stage.

    Examples:
        ResNet-50  -> [3, 4, 6, 3]
        ResNet-101 -> [3, 4, 23, 3]
        ResNet-152 -> [3, 8, 36, 3]
    """

    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()

        # Number of channels produced by the initial convolution (conv1).
        # This value is updated after each stage to match the output
        # dimensionality of the previous Bottleneck.
        self.in_channels = 64

        # ---------------------------------------------------------
        # Initial convolution (conv1 in the paper)
        #
        # Input:
        #   224 × 224 × 3
        #
        # Output:
        #   112 × 112 × 64
        #
        # Uses:
        #   - 64 filters
        #   - 7×7 kernel
        #   - stride = 2
        # ---------------------------------------------------------
        self.conv1 = nn.Conv2d(
            image_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3
        )

        # Batch Normalization after conv1.
        self.bn1 = nn.BatchNorm2d(64)

        # Activation function.
        self.relu = nn.ReLU()

        # Initial max pooling layer.
        #
        # Reduces the spatial resolution:
        #
        # 112×112 -> 56×56
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        # =========================================================
        # ResNet stages (Table 1 of the paper)
        #
        # Paper              Code
        # -----------------------------
        # conv2_x  ------->  layer1
        # conv3_x  ------->  layer2
        # conv4_x  ------->  layer3
        # conv5_x  ------->  layer4
        #
        # Each stage is composed of several Bottleneck blocks.
        #
        # The 'layers' list specifies how many Bottlenecks
        # each stage contains.
        # =========================================================

        # conv2_x
        #
        # Output:
        # 56×56×256
        #
        # Repeated layers[0] times
        # (3 times for ResNet-50).
        self.layer1 = self._make_layer(
            block,
            layers[0],
            out_channels=64,
            stride=1
        )

        # conv3_x
        #
        # First Bottleneck performs downsampling.
        #
        # Output:
        # 28×28×512
        #
        # Repeated layers[1] times
        # (4 times for ResNet-50).
        self.layer2 = self._make_layer(
            block,
            layers[1],
            out_channels=128,
            stride=2
        )

        # conv4_x
        #
        # Output:
        # 14×14×1024
        #
        # Repeated layers[2] times
        # (6 times for ResNet-50).
        self.layer3 = self._make_layer(
            block,
            layers[2],
            out_channels=256,
            stride=2
        )

        # conv5_x
        #
        # Output:
        # 7×7×2048
        #
        # Repeated layers[3] times
        # (3 times for ResNet-50).
        self.layer4 = self._make_layer(
            block,
            layers[3],
            out_channels=512,
            stride=2
        )

        # Global Average Pooling.
        #
        # Converts:
        #
        # 7×7×2048
        #
        # into
        #
        # 1×1×2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Final fully connected classifier.
        #
        # 2048 features -> num_classes
        #
        # (2048 = 512 × expansion)
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):

        # Initial feature extraction.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual stages.
        #
        # conv2_x -> conv3_x -> conv4_x -> conv5_x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global Average Pooling.
        x = self.avgpool(x)

        # Flatten:
        #
        # (N,2048,1,1)
        #
        # ->
        #
        # (N,2048)
        x = x.reshape(x.shape[0], -1)

        # Final classification layer.
        x = self.fc(x)

        return x


    def forward_debug(self, x):

        activations = {}

        # Initial feature extraction.
        x = self.conv1(x)
        activations["conv1"] = x.detach()

        x = self.bn1(x)
        activations["bn1"] = x.detach()

        x = self.relu(x)
        activations["relu"] = x.detach()

        x = self.maxpool(x)
        activations["maxpool"] = x.detach()

        # Residual stages.
        x = self.layer1(x)
        activations["layer1"] = x.detach()

        x = self.layer2(x)
        activations["layer2"] = x.detach()

        x = self.layer3(x)
        activations["layer3"] = x.detach()

        x = self.layer4(x)
        activations["layer4"] = x.detach()

        # Global Average Pooling.
        x = self.avgpool(x)
        activations["avgpool"] = x.detach()

        # Flatten.
        x = x.reshape(x.shape[0], -1)
        activations["flatten"] = x.detach()

        # Final classification layer.
        x = self.fc(x)
        activations["fc"] = x.detach()

        return x, activations


    def _make_layer(self, block, num_residual_block, out_channels, stride):
        """
        Builds one stage of the ResNet architecture.

        According to Table 1 of the ResNet paper, each stage is made of
        multiple Bottleneck blocks.

        Example (ResNet-50):

            conv2_x : [Bottleneck] × 3
            conv3_x : [Bottleneck] × 4
            conv4_x : [Bottleneck] × 6
            conv5_x : [Bottleneck] × 3

        This function constructs one of these stages.
        """

        # By default, the shortcut is an identity mapping.
        # A projection shortcut (1×1 Conv + BatchNorm) is created only
        # if the input and output tensors have different dimensions.
        identity_downsample = None

        # List that will contain all Bottleneck blocks of the stage.
        layers = []

        # -------------------------------------------------------------
        # Check whether the shortcut needs a projection.
        #
        # A projection is required if:
        #
        # 1. stride = 2
        #    -> the spatial resolution changes
        #       (56×56 -> 28×28, etc.)
        #
        # 2. the number of channels changes
        #    (256 -> 512, 512 -> 1024, ...)
        #
        # Without this projection, the residual branch and the shortcut
        # would have different shapes and could not be added together.
        # -------------------------------------------------------------
        if stride != 1 or self.in_channels != out_channels * 4:

            identity_downsample = nn.Sequential(

                # Adjust both spatial resolution and channel dimension
                # of the shortcut branch.
                nn.Conv2d(
                    self.in_channels,
                    out_channels * 4,
                    kernel_size=1,
                    stride=stride
                ),

                nn.BatchNorm2d(out_channels * 4)
            )

        # -------------------------------------------------------------
        # First Bottleneck of the stage.
        #
        # This is the only block that may:
        #
        # - downsample the feature map (stride = 2)
        # - increase the number of channels
        # - use a projection shortcut
        #
        # Example (conv3_x):
        #
        # Input:
        #   56×56×256
        #
        # Output:
        #   28×28×512
        # -------------------------------------------------------------
        layers.append(
            block(
                self.in_channels,
                out_channels,
                identity_downsample,
                stride
            )
        )

        # After the first Bottleneck, every following block receives
        # the expanded number of channels.
        #
        # Example:
        #
        # out_channels = 128
        # expansion = 4
        #
        # output channels = 512
        self.in_channels = out_channels * 4

        # -------------------------------------------------------------
        # Remaining Bottleneck blocks.
        #
        # The paper writes:
        #
        #     [ Bottleneck ] × N
        #
        # The first Bottleneck has already been created above.
        #
        # Therefore, we only need to create the remaining (N - 1)
        # Bottleneck blocks.
        #
        # These blocks DO NOT:
        #
        # - change the spatial resolution
        # - change the number of channels
        #
        # Consequently:
        #
        # - stride = 1
        # - identity shortcut is enough
        # -------------------------------------------------------------
        for i in range(num_residual_block - 1):

            layers.append(
                block(
                    self.in_channels,
                    out_channels
                )
            )

        # Pack all Bottleneck blocks into one Sequential module.
        #
        # Example:
        #
        # conv4_x =
        #
        # Bottleneck ->
        # Bottleneck ->
        # Bottleneck ->
        # Bottleneck ->
        # Bottleneck ->
        # Bottleneck
        #
        return nn.Sequential(*layers)



def ResNet50(img_channels=3, num_classes=1000):
  return ResNet(block, [3,4,6,3], img_channels, num_classes)

def ResNet101(img_channels=3, num_classes=1000):
  return ResNet(block, [3,4,23,3], img_channels, num_classes)

def ResNet152(img_channels=3, num_classes=1000):
  return ResNet(block, [3,8,36,3], img_channels, num_classes)

!kaggle datasets download -d assemelqirsh/chest-x-ray-dataset

!unzip chest-x-ray-dataset.zip

import os
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

DATASET_PATH = "chest_xray"

print(os.listdir(DATASET_PATH))
print(os.listdir(os.path.join(DATASET_PATH, "train")))
def count_images(folder):
    count = 0
    for path in os.listdir(folder):
      if os.path.isfile(os.path.join(folder, path)):
          count += 1
    print('File count:', count)

set = ['test', 'train','val']
type = ['NORMAL', 'PNEUMONIA']
for s in set:
  for t in type:
    print(DATASET_PATH+'/'+s+'/'+t)
    count_images(DATASET_PATH+'/'+s+'/'+t)

img = Image.open("/content/chest_xray/test/NORMAL/IM-0028-0001.jpeg")
print(img.size)
print(img.mode)
plt.imshow(img, cmap="gray")
plt.axis("off")

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(
    root=os.path.join(DATASET_PATH, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(DATASET_PATH, "val"),
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    root=os.path.join(DATASET_PATH, "test"),
    transform=test_transform
)

train_dataset.classes
train_dataset.class_to_idx

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

import torch.optim as optim


model = ResNet50(num_classes=2)
# Sposta il modello su CPU/GPU
model = model.to(device)

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Numero di epoche
num_epochs = 10

for epoch in range(num_epochs):

    #######################
    # TRAIN
    #######################
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    #######################
    # VALIDATION
    #######################
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = correct / total

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_acc:.4f}"
    )

model.eval()

image, label = test_dataset[0]



plt.imshow(image.permute(1, 2, 0))
plt.axis("off")
plt.title(f"True label: {test_dataset.classes[label]}")
plt.axis("off")
plt.show()

image = image.unsqueeze(0).to(device)

with torch.no_grad():
    output, activations = model.forward_debug(image)

print("Logits:", output)

prob = torch.softmax(output, dim=1)
print("probability:", prob)

pred = prob.argmax(dim=1).item()

print("Predicted:", test_dataset.classes[pred])
print("Label:", test_dataset.classes[label])

for layer_name, feat in activations.items():


    if feat.ndim != 4:
        continue

    feat = feat[0].cpu()

    n_maps = min(24, feat.shape[0])
    fig, axes = plt.subplots(4, 6, figsize=(8, 8))
    fig.suptitle(
        f"{layer_name} - shape {tuple(activations[layer_name].shape)}",
        fontsize=14
    )

    for i, ax in enumerate(axes.flat):
        if i < n_maps:
            ax.imshow(feat[i], cmap="gray")
            ax.set_title(f"F{i}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.show()