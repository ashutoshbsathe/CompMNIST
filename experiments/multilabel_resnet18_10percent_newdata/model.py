from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

name_to_model = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
}

class ComplexMNISTModel(nn.Module):
    def __init__(self, base_net, num_feat=512, num_classes=10):
        super().__init__()
        self.num_boxes = 32
        self.base_net = name_to_model[base_net](num_classes=num_feat)
        self.component = nn.Linear(num_feat, num_classes)
        self.composite = nn.Linear(num_feat, num_classes)
    
    def forward(self, x):
        feat = self.base_net(x)
        component = nn.functional.sigmoid(self.component(feat))
        composite = self.composite(feat)
        return composite, component

def test():
    model = ComplexMNISTModel('resnet18')
    from torchsummary import summary
    summary(model.cuda(), (3, 224, 224))

if __name__ == '__main__':
    test()