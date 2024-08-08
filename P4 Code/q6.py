import Testing
import NeuralNetUtil
import NeuralNet


print("----------------- testPenData --------------------")
numNeurons = 0
while numNeurons <= 40:
    print(f"--------------running with {numNeurons} neurons per hidden layer------------------")
    i = 0
    acclist = []
    while i < 5:
        print(f"running iteration # {i + 1}")
        nnet, testAccuracy = Testing.buildNeuralNet(Testing.penData, maxItr=200, hiddenLayerList=[numNeurons])
        acclist.append(testAccuracy)
        i += 1

    print("Iterations finished")
    print(f"accuracy average: {Testing.average(acclist)}")
    print(f"accuracy standard deviation: {Testing.stDeviation(acclist)}")
    print(f"max accuracy: {max(acclist)}")
    numNeurons += 5


print("----------------- testCarData --------------------")
numNeurons = 0
while numNeurons <= 40:
    print(f"--------------running with {numNeurons} neurons per hidden layer------------------")
    i = 0
    acclist = []
    while i < 5:
        print(f"running iteration # {i + 1}")
        nnet, testAccuracy = Testing.buildNeuralNet(Testing.carData, maxItr=200, hiddenLayerList=[numNeurons])
        acclist.append(testAccuracy)
        i += 1

    print("Iterations finished")
    print(f"accuracy average: {Testing.average(acclist)}")
    print(f"accuracy standard deviation: {Testing.stDeviation(acclist)}")
    print(f"max accuracy: {max(acclist)}")
    numNeurons += 5
