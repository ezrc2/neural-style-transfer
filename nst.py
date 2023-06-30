import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import matplotlib.pyplot as plt
#from tqdm import tqdm
import streamlit as st


# https://arxiv.org/pdf/1508.06576.pdf
# https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# https://www.youtube.com/watch?v=imX4kSKDY7s
# https://avandekleut.github.io/nst/

device = 'cpu'
if torch.backends.mps.is_available(): # apple arm gpu
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'

loader = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Lambda(lambda x: x.unsqueeze(0).to(device)) # batch size
])

to_output = transforms.Compose([
    transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
    transforms.Lambda(lambda x: x.squeeze().cpu()),
    transforms.ToPILImage()
])

def gram_matrix(features):
    _, channels, height, width = features.size()
    features = features.view(channels, height * width)
    return features @ features.T

class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        self.vgg19 = models.vgg19(weights='DEFAULT').eval().to(device)
        
    def forward(self, x):
        features = []
        for layer in self.vgg19.features:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                features.append(x)
        return features
    
def fit(style_image, content_image, style_weight=0.01, content_weight=100,
              style_layers=[0, 1, 2, 3, 4], content_layers=[4], epochs=800, lr=0.1):
    model = Vgg19()
    content_features = model(content_image) # compute features of content image
    style_features = model(style_image) # compute features of style image
    
    generated = 0.1 * torch.randn_like(style_image, device=device) # generated, white noise image
    generated.requires_grad_(True)

    optimizer = optim.Adam([generated], lr=lr)
    
    progress_bar = st.progress(0, text='Epochs: ')

    outputs = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        generated_features = model(generated) # compute features of generated image
        
        content_loss = 0
        for layer in content_layers:
            generated_content = generated_features[layer]
            content_target = content_features[layer].detach()
            content_loss += ((generated_content - content_target) ** 2).sum()

        style_loss = 0
        for layer in style_layers:
            generated_style = gram_matrix(generated_features[layer])
            style_target = gram_matrix(style_features[layer]).detach()
            style_loss += ((generated_style - style_target) ** 2).sum()

        # loss between generated image and content image
        loss = (content_weight * content_loss / len(content_layers)) + (style_weight * style_loss / len(style_layers))
    
        loss.backward()
        optimizer.step()

        progress_bar.progress((epoch+1)/epochs, text=f'Epochs: {epoch+1}/{epochs}')

        if (epoch+1) % (epochs // 5) == 0:
            outputs.append(to_output(generated))
        
    return outputs