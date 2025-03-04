from train import model_student
from vec2data import vec2data

import cv2 as cv
import torch
import matplotlib.pyplot as plt


# Load the image
img = cv.imread(r"requirements\imgs\dataset-card.jpg")

# Resize image to (128, 128)
img = cv.resize(img, (128, 128))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Convert to float32 for proper normalization
img_float = torch.tensor(img,dtype=torch.float32)

# Predefined mean and std (for ImageNet-like normalization)
mean = torch.tensor([0.5, 0.5, 0.5])
std = torch.tensor([0.5, 0.5, 0.5])

# Normalize image to [0, 1] range
img_normalized = (img_float / 255.0)

# Normalize with mean and std for each channel
img_normalized = (img_normalized - mean) / std

def draw_again(img, model1, model2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = torch.unsqueeze(img.view(3, 128, 128),dim=0).to(device)  # 🛠 Reshape ve GPU'ya taşı
    model_student = model1.to(device)
    reversed_m = model2.to(device)

    with torch.no_grad():  # 🛠 Gradient takibini kapattık
        student = model_student(img)
        resim = reversed_m(student)

    img_r = resim[0].permute(1,2,0).cpu().numpy() # 🛠 GPU'dan CPU'ya

    fig, axes = plt.subplots(figsize=(10, 5))
    axes.imshow(img_r)
    axes.axis("off")
    plt.show()

draw_again(img_normalized,model_student,vec2data)
