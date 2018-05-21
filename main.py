from nn import NeuralNetwork

nn = NeuralNetwork(784, 100, 10)

def normalizeInputs(inputs):
    return [num / 255 for num in inputs]

print('TRAIN')
with open('mnist_train.csv') as train_file:
    i = 0
    for str in train_file:
        if i == 30000: break
        if i % 1000 == 0: print('train {}'.format(i))
        i += 1

        data = [int(char) for char in str.split(',')]
        inputs = normalizeInputs(data[1:])
        targets = [0.9 if i == data[0] else 0.1 for i in range(10)]
        nn.train(inputs, targets)

print('TEST')
with open('mnist_test.csv') as test_file:
    total = 0
    errors = 0
    for str in test_file:
        data = [int(char) for char in str.split(',')]
        inputs = normalizeInputs(data[1:])
        outputs = nn.predict(inputs)
        result = outputs.index(max(outputs))
        target = data[0]

        if result != target:
            errors += 1
        total += 1

print(total, errors / total * 100)
