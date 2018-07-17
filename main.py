from nn import NeuralNetwork
import random
import time

def normalizeInputs(inputs):
    return [num / 255 for num in inputs]

nn = NeuralNetwork(784, 200, 10)
nn.load('weights60000_100hidden.json')

start = time.time()
with open('mnist_train.csv') as train_file:
    print('TRAIN')
    i = 0
    for str in train_file:
        # if i > 2000: break
        if i % 1000 == 0: print('train {}'.format(i))
        i += 1

        data = [int(char) for char in str.split(',')]
        inputs = normalizeInputs(data[1:])
        targets = [0.9 if i == data[0] else 0.1 for i in range(10)]
        nn.train(inputs, targets)

with open('mnist_test.csv') as test_file:
    print('TEST')
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

    print('Tested: {0}\nErrors: {1:.2f}%'.format(total, errors / total * 100))


end = time.time()
print('Time {} seconds'.format(end - start))

# print(nn.weights_ho)

answer = input('Save weights? (y/n) ').lower()
if answer == 'yes' or answer == 'y':
    nn.save('weights60000_100hidden.json')
    print('Weights has been saved')
else:
    print('Wegihts has been dropped')

