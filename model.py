import torch
from torch import nn
import torchvision
import torchvision.transforms as tf
from PIL import Image
from countries import countries


class Baseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, image):
        prediction = self.resnet(image)
        num_locations = 55
        return prediction[:, :num_locations]


def guess(file):
    # Transform Input Image
    transforms = [tf.ToTensor(), tf.Resize((256, 256), antialias=None)]
    transforms = tf.Compose(transforms)
    image = transforms(Image.open(file))[:3]

    # Load in Model
    model = Baseline()
    model.eval()
    state_dict = torch.load("./resnet50-steps55k.ckpt",
                            map_location=torch.device('cpu'))['state_dict']
    d = {}
    for k in state_dict:
        d[k.replace("model.", "")] = state_dict[k]
    model.load_state_dict(d)

    country_preds = []
    # Create Prediction
    with torch.no_grad():
        for (i, pred) in enumerate(model(image[None]).tolist()[0]):
            country_preds.append((pred, countries[i]))
    country_preds = sorted(country_preds, key=lambda x: x[0], reverse=True)

    return { str(i):
        { 
            'prob': country_preds[i][0], 
            'name': country_preds[i][1] 
        } for i in range(len(country_preds)) 
    }