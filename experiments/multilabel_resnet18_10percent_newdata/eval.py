import collections
import torch 
from torch import nn
import torchvision
import numpy as np 
from model import ComplexMNISTModel
from complex_mnist import ComplexMNISTDataset
from tqdm import tqdm
torch.manual_seed(161803398)
np.random.seed(161803398)
ckpt_path = './saved_models/checkpoint_epoch_12_total_loss_0.011851892806589603.ckpt'

ckpt = torch.load(ckpt_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ComplexMNISTModel('resnet18').to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
test = ComplexMNISTDataset(root='../../new_mnist_complex/', transform=torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor()
]), dataset_type='test')
testloader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False)
bce = nn.BCELoss()
xent = nn.CrossEntropyLoss()
pbar = tqdm(testloader, desc='test')
total_samples = 0
total_xent = 0
total_bce = 0
total_correct_composite = 0
total_correct_components = 0
for i, batch in enumerate(pbar):
    images, composite, component = batch 
    images, composite, component = images.to(device), composite.to(device), component.to(device)

    pred_composite, pred_component = model(images)
    xent_loss = xent(pred_composite, composite.long())
    bce_loss = bce(pred_component, component.float())

    total_xent += images.size(0) * xent_loss.item()
    total_bce += images.size(0) * bce_loss.item()
    total_samples += images.size(0)

    total_correct_composite += (torch.max(pred_composite, 1)[1] == composite).sum().item()
    total_correct_components += ((pred_component >= 0.5).long() == component.long()).sum().item()
    pbar.set_description('test: xent_loss: {:.2f} bce_loss: {:.2f} total_loss: {:.2f}'.format(
        total_xent / total_samples,
        total_bce / total_samples,
        (total_xent + 5 * total_bce) / total_samples
    ))
print(total_correct_composite, '/', total_samples)
print(total_correct_components, '/', total_samples * 10)