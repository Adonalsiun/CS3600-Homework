import Testing
import NeuralNetUtil
import NeuralNet

numNeurons = 0
while True:
    print(f"--------------running with {numNeurons} neurons per hidden layer------------------")
    i = 0
    acclist = []
    while i < 5:
        print(f"running iteration # {i + 1}")
        nnet, testAccuracy = NeuralNet.buildNeuralNet(Testing.xorData, maxItr=200, hiddenLayerList=[numNeurons])
        acclist.append(testAccuracy)
        i += 1

    print("Iterations finished")
    print(f"accuracy average: {Testing.average(acclist)}")
    print(f"accuracy standard deviation: {Testing.stDeviation(acclist)}")
    print(f"max accuracy: {max(acclist)}")
    if Testing.average(acclist) == 1:
        break
    numNeurons += 1
0

def getList(position, length):
    # This function needs to be defined based on what it's supposed to do.
    # For now, I'll assume it generates a list with a 1 at the given position
    # and 0 elsewhere in a list of the given length.
    return [1 if i == position - 1 else 0 for i in range(length)]
def getNNXorData(fileString="datasets/xor.data.txt", limit=100000):
    """
    Returns a limited number of examples from the file passed as a string.
    """
    examples = []
    attrValues = {}
    attrs = ['x', 'y']
    attr_values = [['0', '1'], ['0', '1']]

    attrNNList = [('x', {'0': getList(1, 2), '1': getList(2, 2)}),
                  ('y', {'0': getList(1, 2), '1': getList(2, 2)})]

    classNNList = {'0': [1, 0], '1': [0, 1]}

    for index in range(len(attrs)):
        attrValues[attrs[index]] = attrNNList[index][1]

    lineNum = 0
    with open(fileString) as data:
        for line in data:
            inVec = []
            outVec = []
            count = 0
            for val in line.strip().split(','):
                if count == 2:
                    outVec = classNNList[val]
                else:
                    inVec.append(attrValues[attrs[count]][val])
                count += 1
            examples.append((inVec, outVec))
            lineNum += 1
            if lineNum >= limit:
                break
    return examples

def buildExamplesFromXorData(size=4):
    xorData = getNNXorData()
    xorDataTrainList = []
    for cdRec in xorData:
        tmpInVec = []
        for cdInRec in cdRec[0]:
            tmpInVec.extend(cdInRec)
        tmpList = (tmpInVec, cdRec[1])
        xorDataTrainList.append(tmpList)
    xorDataTestList = xorDataTrainList
    return xorDataTrainList, xorDataTestList