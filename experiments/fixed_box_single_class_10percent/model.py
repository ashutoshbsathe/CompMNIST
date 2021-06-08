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
    def __init__(self, base_net, num_feat=512, num_boxes=32, num_classes=10):
        super().__init__()
        self.num_boxes = 32
        self.base_net = name_to_model[base_net](num_classes=num_feat)
        self.component_boxes = nn.Linear(num_feat, num_boxes * 4)
        self.component_class = nn.Linear(num_feat, num_classes)
        self.composite_class = nn.Linear(num_feat, num_classes)
    
    def forward(self, x):
        feat = self.base_net(x)
        component_boxes = self.component_boxes(feat)
        component_class = self.component_class(feat)
        composite_class = self.composite_class(feat)
        return composite_class, component_boxes, component_class 

def test():
    model = ComplexMNISTModel('resnet18')
    from torchsummary import summary
    summary(model.cuda(), (3, 224, 224))

if __name__ == '__main__':
    test()