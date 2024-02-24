# -*- coding: utf-8 -*-

import pathlib

import numpy as np
import skimage
import torch
import torchvision
import matplotlib.pyplot as plt
import torchxrayvision as xrv

import streamlit as st


#st.set_page_config(layout='wide')
st.title('Анатомическая сегментация')

model = xrv.baseline_models.chestx_det.PSPNet()
model.targets_ru = {
    'Left Clavicle':          'Левая ключица',
    'Right Clavicle':	      'Правая ключица',
    'Left Scapula':	      'Левая лопатка',
    'Right Scapula':          'Правая лопатка',
    'Left Lung':              'Левое легкое',
    'Right Lung':             'Правое легкое',
    'Left Hilus Pulmonis':    '', #'Левый Hilus Pulmonis',
    'Right Hilus Pulmonis':   '', #'Правый Hilus Pulmonis',
    'Heart':                  'Сердце',
    'Aorta':                  'Аорта',
    'Facies Diaphragmatica':  '', #''Фация диафрагмальная',
    'Mediastinum':            '', #''Средостение',
    'Weasand':                '', #''Глотка',
    'Spine':                  'Позвоночник',
}

files = pathlib.Path('data_padchest_small').glob('*.png')
filename = st.selectbox('Выберите файл', [f for f in files])
st.write(f'Файл: {filename}')

#img = skimage.io.imread("../tests/covid-19-pneumonia-58-prior.jpg")
img = skimage.io.imread(filename)
img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
#img = img.mean(2)[None, ...] # Make single color channel
img = img[None, :, :] # works for PNG 8-bit sRGB 256c

transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(512)])
img = transform(img)
img = torch.from_numpy(img)

with torch.no_grad():
    pred = model(img)

# plt.figure(figsize = (26,5))
# plt.subplot(1, len(model.targets) + 1, 1)
# plt.imshow(img[0], cmap='gray')
# for i in range(len(model.targets)):
#     plt.subplot(1, len(model.targets) + 1, i+2)
#     plt.imshow(pred[0, i])
#     plt.title(model.targets[i])
#     plt.axis('off')
# plt.tight_layout()

pred = 1 / (1 + np.exp(-pred))  # sigmoid
#pred[pred < 0.5] = 0
#pred[pred > 0.5] = 1

fig1, ax1 = plt.subplots(figsize = (4*4, 3*4))
plt.subplot(4, 3, 1); plt.axis('off')
plt.subplot(4, 3, 2)
plt.imshow(img[0], cmap='gray')
plt.subplot(4, 3, 3); plt.axis('off')
fig_count = 4
for i in range(len(model.targets)):
    title_ru = model.targets_ru[model.targets[i]]
    if not title_ru:
        continue
    plt.subplot(4, 3, fig_count)
    plt.imshow(img[0], cmap='gray')
    plt.contour(pred[0, i], [0.5], colors=['red'])
    plt.title(title_ru)
    plt.axis('off')
    fig_count += 1
plt.tight_layout()
#plt.subplots_adjust(wspace=0, hspace=0)

st.pyplot(fig1)

st.title('Предсказание патологий')

models = {
#    'densenet121-res224-all': 'All',
#    'densenet121-res224-rsna': 'RSNA Pneumonia Challenge',
    'densenet121-res224-chex': 'CheXpert (Stanford)',
    'densenet121-res224-nih': 'NIH chest X-ray8',
    'densenet121-res224-pc': 'PadChest (University of Alicante)',
#    'densenet121-res224-mimic_nb': 'MIMIC-CXR (MIT)',
#    'densenet121-res224-mimic_ch': 'MIMIC-CXR (MIT)',
}
modname = st.selectbox('Выберите модель', [m for m in models])
model = xrv.models.get_model(modname)
pathologies = {p: model.pathologies.index(p) for p in model.pathologies if p}

img = skimage.io.imread(filename)
img = xrv.datasets.normalize(img, 255)
img = img[None, :, :] # Add color channel

transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
img = transform(img)
img = torch.from_numpy(img).unsqueeze(0)
#img = torch.from_numpy(img)

st.write(f'Модель {modname} ({models[modname]}) может предсказывать следующие патологии:')
st.write(', '.join([p.replace("_", " ") for p in model.pathologies if p]))
pathology = st.selectbox('Выберите патологию для визуализации',
                         [p for p in pathologies])
target = pathologies[pathology]
img = img.requires_grad_()
outputs = model(img)
grads = torch.autograd.grad(outputs[:, target], img)[0][0][0]
blurred = skimage.filters.gaussian(grads.detach().cpu().numpy()**2, sigma=(5, 5), truncate=3.5)

fig2, ax2 = plt.subplots(figsize = (1, 3)) # figsize = (4*4, 3*4)
plt.imshow(img[0][0].detach().cpu().numpy(), cmap='gray', aspect='auto')
plt.imshow(blurred, alpha=0.3)
plt.axis('off')
st.pyplot(fig2)
st.write(outputs[:, target])
