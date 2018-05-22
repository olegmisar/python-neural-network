from nn import NeuralNetwork
from PIL import Image

nn = NeuralNetwork(784, 100, 10)
nn.load('weights60000.json')


with Image.open('images/6.png') as img_file:
    img_file = img_file.convert('L').resize((28, 28), Image.LANCZOS).point(lambda p: 255 - p)
    img_file.show()
    image = img_file.getdata()
    output = nn.predict(image)
    result = output.index(max(output))
    print(result)

