import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd

mobilenet_v2 = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")
mobilenet_v3_small = models.mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.DEFAULT")
mobilenet_v3_large = models.mobilenet_v3_large(weights="MobileNet_V3_Large_Weights.DEFAULT")
alexnet = models.alexnet(weights="AlexNet_Weights.DEFAULT")
#yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
resnet = models.resnet50(weights="ResNet50_Weights.DEFAULT")
efficientnet = models.efficientnet_b0(weights="EfficientNet_B0_Weights.DEFAULT")
vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
ResNet50 = models.resnet50(weights="ResNet50_Weights.DEFAULT")


modellist = [ResNet50] #yolo, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, alexnet,  resnet, efficientnet, vgg, 

# iterate over the models
for model in modellist:
    model.eval()

    # Image normalization
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Image import
    input_image = Image.open('.\src\\tools\\imageClassification\\images\\000002_1.jpg')

    # Preprocess image and prepare batch
    input_tensor  = preprocess(input_image)
    input_batch  = input_tensor.unsqueeze(0)

    # use cuda cores on gpu
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    
    # Print Tensor
    #print(output)
    
    # Save Tensor to file
    #torch.save(output, '.\src\\tools\\imageClassification\\output.t')
    
    # calculate probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0) * 100

    # read the categories
    with open("./imageClassification/imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # Show top 5 categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    print("\n",model._get_name())
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())