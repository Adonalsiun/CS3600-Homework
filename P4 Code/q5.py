import Testing
import NeuralNetUtil
import NeuralNet

print("----------------- testPenData --------------------")
i = 0
acclist = []
while i < 5:
    print(f"running iteration # {i + 1}")
    nnet, testAccuracy = Testing.testPenData()
    acclist.append(testAccuracy)
    i += 1

print("Iterations finished")
print(f"accuracy average: {Testing.average(acclist)}")
print(f"accuracy standard deviation: {Testing.stDeviation(acclist)}")
print(f"max accuracy: {max(acclist)}")

print("----------------- testCarData --------------------")
i = 0
acclist = []
while i < 5:
    print(f"running iteration # {i + 1}")
    nnet, testAccuracy = Testing.testCarData()
    acclist.append(testAccuracy)
    i += 1

print("Iterations finished")
print(f"accuracy average: {Testing.average(acclist)}")
print(f"accuracy standard deviation: {Testing.stDeviation(acclist)}")
print(f"max accuracy: {max(acclist)}")
