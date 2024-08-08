import Testing
import NeuralNetUtil
import NeuralNet

i = 0
acclist = []
while i < 5:
    print(f"running iteration # {i + 1}")
    nnet, testAccuracy = Testing.testExtraData()  # Make sure this function is set up to handle poker hand data
    acclist.append(testAccuracy)
    i += 1

print("Iterations finished")
print(f"accuracy average: {Testing.average(acclist)}")
print(f"accuracy standard deviation: {Testing.stDeviation(acclist)}")
print(f"max accuracy: {max(acclist)}")
