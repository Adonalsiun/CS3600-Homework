import copy
import sys
from datetime import datetime
from math import exp
from random import random, randint, choice

class Perceptron(object):
    """
    Class to represent a single Perceptron in the net.
    """
    def __init__(self, inSize=1, weights=None):
        self.inSize = inSize+1#number of perceptrons feeding into this one; add one for bias
        if weights is None:
            #weights of previous layers into this one, random if passed in as None
            self.weights = [1.0]*self.inSize
            self.setRandomWeights()
        else:
            self.weights = weights

    def getWeightedSum(self, inActs):
        """
        Returns the sum of the input weighted by the weights.

        Inputs:
            inActs (list<float/int>): input values, same as length as inSize
        Returns:
            float
            The weighted sum
        """
        return sum([inAct*inWt for inAct,inWt in zip(inActs,self.weights)])

    def sigmoid(self, value):
        """
        Return the value of a sigmoid function.

        Args:
            value (float): the value to get sigmoid for
        Returns:
            float
            The output of the sigmoid function parametrized by
            the value.
        """
        """YOUR CODE"""
        # Sigmoid function = 1/(1+e^-x) from Textbook pg. 1383, Equation 21.1
        expValue = exp(-value)
        denominator = 1 + expValue
        sigmoidValue = 1 / denominator
        return sigmoidValue

    def sigmoidActivation(self, inActs):
        """
        Returns the activation value of this Perceptron with the given input.
        Same as g(z) in book.
        Remember to add 1 to the start of inActs for the bias input.

        Inputs:
            inActs (list<float/int>): input values, not including bias
        Returns:
            float
            The value of the sigmoid of the weighted input
        """
        """YOUR CODE"""
        # Add 1 to start of list for bias
        # Implementation from Lecture 16, slide 20
        biases = [1] + list(inActs)
        weightedSum = self.getWeightedSum(biases)
        sigmoidValue = self.sigmoid(weightedSum)
        return sigmoidValue

    def sigmoidDeriv(self, value):
        """
        Return the value of the derivative of a sigmoid function.

        Args:
            value (float): the value to get sigmoid for
        Returns:
            float
            The output of the derivative of a sigmoid function
            parametrized by the value.
        """
        """YOUR CODE"""
        # Implementation from Textbook pg. 1267, Equation 19.10
        sigmoidValue = 1 / (1 + exp(-value))
        sigmoidDerivValue = sigmoidValue * (1 - sigmoidValue)
        return sigmoidDerivValue

    def sigmoidActivationDeriv(self, inActs):
        """
        Returns the derivative of the activation of this Perceptron with the
        given input. Same as g'(z) in book (note that this is not rounded.
        Remember to add 1 to the start of inActs for the bias input.

        Inputs:
            inActs (list<float/int>): input values, not including bias
        Returns:
            int
            The derivative of the sigmoid of the weighted input
        """
        """YOUR CODE"""
        # Implementation from Lecture 16, slide 20
        biases = [1] + list(inActs)
        weightedSum = self.getWeightedSum(biases)
        sigmoidDerivValue = self.sigmoidDeriv(weightedSum)
        return sigmoidDerivValue

    def updateWeights(self, inActs, alpha, delta):
        """
        Updates the weights for this Perceptron given the input delta.
        Remember to add 1 to the start of inActs for the bias input.

        Inputs:
            inActs (list<float/int>): input values, not including bias
            alpha (float): The learning rate
            delta (float): If this is an output, then g'(z)*error
                           If this is a hidden unit, then the as defined-
                           g'(z)*sum over weight*delta for the next layer
        Returns:
            float
            Return the total modification of all the weights (sum of each abs(modification))
        """
        totalModification = 0
        """YOUR CODE"""
        # Implementation from Textbook, pg. 1263, Equation 19.8
        biases = [1] + list(inActs)
        alpdel = alpha * delta
        for i in range(len(self.weights)):
            modification = alpdel * biases[i]
            self.weights[i] += modification
            totalModification += abs(modification)
        return totalModification

    def setRandomWeights(self):
        """
        Generates random input weights that vary from -1.0 to 1.0
        """
        for i in range(self.inSize):
            self.weights[i] = (random() + .0001) * (choice([-1,1]))

    def __str__(self):
        """ toString """
        outStr = ''
        outStr += 'Perceptron with %d inputs\n'%self.inSize
        outStr += 'Node input weights %s\n'%str(self.weights)
        return outStr

class NeuralNet(object):
    """
    Class to hold the net of perceptrons and implement functions for it.
    """
    def __init__(self, layerSize):#default 3 layer, 1 percep per layer
        """
        Initiates the NN with the given sizes.

        Args:
            layerSize (list<int>): the number of perceptrons in each layer
        """
        self.layerSize = layerSize #Holds number of inputs and percepetrons in each layer
        self.outputLayer = []
        self.numHiddenLayers = len(layerSize)-2
        self.hiddenLayers = [[] for x in range(self.numHiddenLayers)]
        self.numLayers =  self.numHiddenLayers+1

        #build hidden layer(s)
        for h in range(self.numHiddenLayers):
            for p in range(layerSize[h + 1]):
                percep = Perceptron(layerSize[h]) # num of perceps feeding into this one
                self.hiddenLayers[h].append(percep)

        #build output layer
        for i in range(layerSize[-1]):
            percep = Perceptron(layerSize[-2]) # num of perceps feeding into this one
            self.outputLayer.append(percep)

        #build layers list that holds all layers in order - use this structure
        # to implement back propagation
        self.layers = [self.hiddenLayers[h] for h in range(self.numHiddenLayers)] + [self.outputLayer]

    def __str__(self):
        """toString"""
        outStr = ''
        outStr +='\n'
        for hiddenIndex in range(self.numHiddenLayers):
            outStr += '\nHidden Layer #%d'%hiddenIndex
            for index in range(len(self.hiddenLayers[hiddenIndex])):
                outStr += 'Percep #%d: %s'%(index,str(self.hiddenLayers[hiddenIndex][index]))
            outStr +='\n'
        for i in range(len(self.outputLayer)):
            outStr += 'Output Percep #%d:%s'%(i,str(self.outputLayer[i]))
        return outStr

    def feedForward(self, inActs):
        """
        Propagate input vector forward to calculate outputs.

        Args:
            inActs (list<float>): the input to the NN (an example)
        Returns:
            list<list<float/int>>
            A list of lists. The first list is the input list, and the others are
            lists of the output values of all perceptrons in each layer.
        """
        """YOUR CODE"""
        # Implementation from Lecture 17, Slide 37
        activations = []
        activations.append(inActs)
        for layer in self.layers:
            previousLayer = activations[-1]
            currentLayer = []
            for perceptron in layer:
                activation = perceptron.sigmoidActivation(previousLayer)
                currentLayer.append(activation)
            activations.append(currentLayer)
        return activations

    def backPropLearning(self, examples, alpha):
        """
        Run a single iteration of backward propagation learning algorithm.
        See the text and slides for pseudo code.

        Args:
            examples (list<tuple<list<float>,list<float>>>):
              for each tuple first element is input(feature)"vector" (list)
              second element is output "vector" (list)
            alpha (float): the alpha to training with
        Returns
           tuple<float,float>

           A tuple of averageError and averageWeightChange, to be used as stopping conditions.
           averageError is the summed error^2/2 of all examples, divided by numExamples*numOutputs.
           averageWeightChange is the summed absolute weight change of all perceptrons,
           divided by the sum of their input sizes (the average weight change for a single perceptron).
        """
        #keep track of output
        averageError = 0
        averageWeightChange = 0
        numWeights = 0

        for example in examples:#for each example
            #keep track of deltas to use in weight change
            deltas = []
            #Neural net output list
            # Store output layer in allLayerOutput, then lastLayerOutput takes the prediction
            allLayerOutput = self.feedForward(example[0]) # """FILL IN - neural net output list computation"""
            lastLayerOutput = allLayerOutput[-1]
            #Empty output layer delta list
            outDelta = []
            #iterate through all output layer neurons
            for outputNum in range(len(example[1])):
                # G' is the derivative of the activation function, use allLayerOutput
                gPrime = self.outputLayer[outputNum].sigmoidActivationDeriv(allLayerOutput[-2]) # """FILL IN"""
                error = example[1][outputNum] - lastLayerOutput[outputNum] # """FILL IN - error for this neuron"""
                delta = error * gPrime # """FILL IN - delta for this neuron"""
                averageError+=error*error/2
                outDelta.append(delta)
            deltas.append(outDelta)

            """
            Backpropagate through all hidden layers, calculating and storing
            the deltas for each perceptron layer.
            """
            for layerNum in range(self.numHiddenLayers-1,-1,-1):
                layer = self.layers[layerNum]
                nextLayer = self.layers[layerNum+1]
                hiddenDelta = []
                #Iterate through all neurons in this layer
                for neuronNum in range(len(layer)):
                    # Compute delta based on next layer, so iterate through that and multiply by gprime
                    gPrime = layer[neuronNum].sigmoidActivationDeriv(allLayerOutput[layerNum]) # """FILL IN"""
                    delta = gPrime * sum([nextLayer[x].weights[neuronNum + 1] * deltas[0][x] for x in range(len(nextLayer))]) # """FILL IN - delta for this neuron
                               # Carefully look at the equation here,
                                # it is easy to do this by intuition incorrectly"""
                    hiddenDelta.append(delta)
                deltas = [hiddenDelta]+deltas

            """
            Having aggregated all deltas, update the weights of the
            hidden and output layers accordingly.
            """
            for numLayer in range(0,self.numLayers):
                layer = self.layers[numLayer]
                for numNeuron in range(len(layer)):
                    weightMod = layer[numNeuron].updateWeights(allLayerOutput[numLayer], alpha, deltas[numLayer][numNeuron]) # """FILL IN"""
                    averageWeightChange += weightMod
                    numWeights += layer[numNeuron].inSize
            #end for each example
        #calculate final output
        averageError /= (len(examples)*len(examples[0][1]))             #number of examples x length of output vector
        averageWeightChange/=(numWeights)
        return averageError, averageWeightChange

def buildNeuralNet(examples, alpha=0.1, weightChangeThreshold = 0.00008,hiddenLayerList = [1], maxItr = sys.maxsize, startNNet = None):
    """
    Train a neural net for the given input.

    Args:
        examples (tuple<list<tuple<list,list>>,
                        list<tuple<list,list>>>): A tuple of training and test examples
        alpha (float): the alpha to train with
        weightChangeThreshold (float):           The threshold to stop training at
        maxItr (int):                            Maximum number of iterations to run
        hiddenLayerList (list<int>):             The list of numbers of Perceptrons
                                                 for the hidden layer(s).
        startNNet (NeuralNet):                   A NeuralNet to train, or none if a new NeuralNet
                                                 can be trained from random weights.
    Returns
       tuple<NeuralNet,float>

       A tuple of the trained Neural Network and the accuracy that it achieved
       once the weight modification reached the threshold, or the iteration
       exceeds the maximum iteration.
    """
    examplesTrain,examplesTest = examples
    numIn = len(examplesTrain[0][0])
    numOut = len(examplesTest[0][1])
    time = datetime.now().time()
    if startNNet is not None:
        hiddenLayerList = [len(layer) for layer in startNNet.hiddenLayers]
    print ("Starting training at time %s with %d inputs, %d outputs, %s hidden layers, size of training set %d, and size of test set %d"\
                                                    %(str(time),numIn,numOut,str(hiddenLayerList),len(examplesTrain),len(examplesTest)))
    layerList = [numIn]+hiddenLayerList+[numOut]
    nnet = NeuralNet(layerList)
    if startNNet is not None:
        nnet =startNNet
    """
    YOUR CODE
    """
    iteration=0
    trainError=0
    weightMod=0

    """
    Iterate for as long as it takes to reach weight modification threshold
    """
        #if iteration%10==0:
        #    print '! on iteration %d; training error %f and weight change %f'%(iteration,trainError,weightMod)
        #else :
        #    print '.',

    condition = True
    while condition:
        trainError, weightMod = nnet.backPropLearning(examplesTrain, alpha)
        iteration += 1
        if iteration % 10 == 0:
            print
            '! on iteration %d; training error %f and weight change %f' % (iteration, trainError, weightMod)
        else:
            print
            '.',

        condition = iteration < maxItr and weightMod > weightChangeThreshold


    time = datetime.now().time()
    print ('Finished after %d iterations at time %s with training error %f and weight change %f'%(iteration,str(time),trainError,weightMod))

    """
    Get the accuracy of your Neural Network on the test examples.
    For each text example, you should first feedforward to get the NN outputs. Then, round the list of outputs from the output layer of the neural net.
    If the entire rounded list from the NN matches with the known list from the test example, then add to testCorrect, else add to  testError.
    """

    testError = 0
    testCorrect = 0

    results = []

    for test in examplesTest:
        output = nnet.feedForward(test[0])
        roundedOutput = [round(val) for val in output[-1]]
        results.append(roundedOutput == test[1])

    testCorrect = sum(results)
    testAccuracy = testCorrect / len(examplesTest)

    print('Feed Forward Test correctly classified %d, incorrectly classified %d, test accuracy %f\n'%(testCorrect,testError,testAccuracy))

    """return something"""
    return nnet, testAccuracy