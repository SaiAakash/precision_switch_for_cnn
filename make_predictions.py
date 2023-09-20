import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from conv_net import Net

def predict(model, img):
    with torch.no_grad():
        output = model(img)
    prediction_label = output.argmax(dim=1, keepdim=True) 
    return prediction_label

mnist_classifier = Net()
mnist_classifier.load_state_dict(torch.load('mnist_cnn.pt'))

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

test_data = datasets.MNIST('../data', train=False,
                       transform=transform)

testloader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = True)
test_img, true_label = next(iter(testloader))
prediction = predict(mnist_classifier, test_img)
print("True Label", true_label)
print("Prediction Label", prediction)

fig = plt.figure()
plt.imshow(torch.squeeze(test_img, 0).numpy()[0], cmap = 'gray')
plt.show()