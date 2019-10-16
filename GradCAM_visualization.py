"""
Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization
https://arxiv.org/pdf/1610.02391.pdf

My goal is to highlight where ResNet50 trained on ImageNet  is looking to make 
the correct dog classification using the visualization idea of Grad-CAM.
"""
import urllib

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms


DOG_CLASS = 258

# Load an ImageNet trained model
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model.eval()

# Get the feature extraction part of the model
model_conv = nn.Sequential(*list(model.children())[:-2])
for param in model_conv.parameters():
    param.requires_grad = False

# Get the classification part of the model
model_clf  = nn.Sequential(
    list(model.children())[-2],
    nn.Flatten(),
    list(model.children())[-1]
)
for param in model_clf:
    param.requires_grad = True


# Download an example image from the pytorch website
url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
try: 
    urllib.URLopener().retrieve(url, filename)
except: 
    urllib.request.urlretrieve(url, filename)


# Prepare the image
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model


# Grad-CAM part
with torch.no_grad():
    features_maps = model_conv(input_batch)
    features_maps = nn.Parameter(features_maps, requires_grad=True)
output = model_clf(features_maps)


# Only on the first image
y_c = output[:, [DOG_CLASS]]# Score for class `c` before softmax

y_c.mean().backward()



features_maps_importance = features_maps.grad.mean(2).mean(2)
features_maps_importance = features_maps_importance.unsqueeze(-1).unsqueeze(-1)

heat_map = torch.relu((features_maps_importance * features_maps).mean(1))
heat_map = heat_map.unsqueeze(1)

upsampler = nn.UpsamplingBilinear2d(size=224)
upsampled_heat_map = upsampler(heat_map)


# Display Result
def unnormalize(batch, mean, std):
    for t, m, s in zip(batch, mean, std):
        t.mul_(s).add_(m)
    return batch

unnormalized_input_batch = unnormalize(
    input_batch.permute(0, 2, 3, 1),
    preprocess.transforms[-1].mean,
    preprocess.transforms[-1].std
)

plt.title("Interest Zone - GradCAM coarse localization")
plt.imshow(unnormalized_input_batch[0].detach().numpy())
plt.imshow(upsampled_heat_map[0, 0].detach().numpy(), alpha=.7)
plt.show()


