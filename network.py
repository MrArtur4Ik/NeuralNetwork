import numpy as np

sigmoid = lambda x: 1/(1+np.exp(-x))
mse_loss = lambda y_true, y_pred: np.mean((y_true - y_pred)**2) #Средняя квадратичная ошибка
sigmoid_dx = lambda x: x*(1-x) #Производная сигмоиды

class Neuron:
	def __init__(self, weights, bias=0):
		self.weights = np.array(weights)
		self.bias = float(bias)
	
	def feedforward(self, inputs):
		return sigmoid(np.dot(self.weights, np.array(inputs)) + self.bias)

class NeuralNetwork:
	layers = [] #Не считая входного слоя
	def __init__(self, layers: list, inputs_number: int):
		self.layers = layers
		self.inputs_number = inputs_number
	
	def feedforward_all(self, inputs):
		inputs = np.array(inputs)
		outputs = [inputs]
		for layer in self.layers:
			new_inputs = []
			for neuron in layer:
				new_inputs.append(neuron.feedforward(inputs))
			inputs = new_inputs
			outputs.append(np.array(new_inputs))
		return outputs

	def feedforward(self, inputs):
		for layer in self.layers:
			new_inputs = []
			for neuron in layer:
				new_inputs.append(neuron.feedforward(inputs))
			inputs = new_inputs
		return np.array(inputs)
	
	'''Back propogation'''
	def backpropogation(self, inputs, expected_outputs, learning_rate=0.1):
		outputs = self.feedforward_all(inputs)
		output = outputs.pop()
		gradients = (output-np.array(expected_outputs))*sigmoid_dx(output)
		layer_count = len(self.layers)
		for layer in self.layers[::-1]:
			for i, neuron in enumerate(layer):
				gradient = gradients[i]
				neuron.weights = neuron.weights-outputs[-1:][0]*gradient*learning_rate
				neuron.bias = neuron.bias-gradient*learning_rate
				#a[-1:][0] берёт последний элемент массива
				'''for j, weight in enumerate(neuron.weights):
					neuron.weights[j] = weight-outputs[-1:][0][j]*gradient*learning_rate'''
			if layer_count > 1:
				new_gradients = []
				for i, out in enumerate(outputs[-1:][0]):
					new_gradients.append(np.dot([weights[i] for weights in [neuron.weights for neuron in layer]], gradients) * \
						sigmoid_dx(out))
				gradients = new_gradients
				output = outputs.pop()
			layer_count -= 1

def network_to_json(network: NeuralNetwork):
	a = []
	for layer in network.layers:
		b = []
		for neuron in layer:
			b.append([neuron.bias] + list(neuron.weights))
		a.append(b)
	return {"inputs_number": network.inputs_number, "weights": a}

def json_to_network(json: list):
	layers = []
	for layer in json["weights"]:
		neurons = []
		for neuron in layer:
			neurons.append(Neuron(neuron[1:], neuron[0]))
		layers.append(neurons)
	return NeuralNetwork(layers, json["inputs_number"])

def generate_random_network(neurons_count: list, weights_range: tuple = (-1, 1)):
	rand = lambda count: [np.random.uniform(*weights_range) for i in range(count)]
	layers = []
	c = neurons_count[0]
	for count in neurons_count[1:]:
		layers.append([Neuron(rand(c)) for i in range(count)])
		c = count
	return NeuralNetwork(layers, neurons_count[0])
