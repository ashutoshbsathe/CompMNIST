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
ckpt_path = './saved_models\checkpoint_epoch_14_total_loss_0.0012445696629583836.ckpt'

ckpt = torch.load(ckpt_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ComplexMNISTModel('resnet18').to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
test = ComplexMNISTDataset(root='../new_mnist_complex/', transform=torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor()
]), dataset_type='test')
testloader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False)
boxloss = nn.MSELoss()
xent = nn.CrossEntropyLoss()
pbar = tqdm(testloader, desc='train')
total_samples = 0
total_xent = 0
total_boxes = 0
total_component = 0
# table[0] = both wrong 
# table[1] = composite incorrect component correct
# table[2] = composite correct component incorrect
# table[3] = both correct
table = [0, 0, 0, 0]
for i, batch in enumerate(pbar):
    images, composite_class, component_boxes, component_class = batch 
    images, composite_class, component_boxes, component_class = \
        images.to(device), composite_class.to(device), component_boxes.to(device), \
        component_class.to(device)
    pred_composite, pred_boxes, pred_component = model(images)
    xent_loss = xent(pred_composite, composite_class.long())
    boxes = boxloss(pred_boxes, component_boxes.float())
    component_loss = xent(pred_component, component_class)
    total_xent += images.size(0) * xent_loss.item()
    total_boxes += images.size(0) * boxes.item()
    total_component += images.size(0) * component_loss.item()
    total_samples += images.size(0)

    bool_correct_composite = (torch.max(pred_composite, 1)[1] == composite_class.long())
    bool_correct_components = (torch.max(pred_component, 1)[1] == component_class.long())

    def logical_not(x):
        return 1 - x 
    
    table[0] += (logical_not(bool_correct_composite) & logical_not(bool_correct_components)).sum()
    table[1] += (logical_not(bool_correct_composite) & bool_correct_components).sum()
    table[2] += (bool_correct_composite & logical_not(bool_correct_components)).sum()
    table[3] += (bool_correct_composite & bool_correct_components).sum()
    pbar.set_description('test: xent_loss: {:.2f} boxloss: {:.2f} component_loss: {:.2f} total_loss: {:.2f}'.format(
        total_xent / total_samples,
        total_boxes / total_samples,
        total_component / total_samples,
        (total_xent + 1/32 * total_boxes + total_component) / total_samples
    ))

print(table)