import os

codePath = '/Users/bytedance/codebase/'
proSet = set(x + 1 for x in range(10))

# Map to projects
projDict = [
    '1-tomcat/',
    '2-gradle/',
    '3-hadoop/',
    '4-spring-framework/',
    '5-zxing/',
    '6-cassandra/',
    '7-fresco/',
    '8-guava/',
    '9-kafka/',
    '10-wildfly/'
]

# Traverse all the files/dirs in the given path
def listdir(path):
    files = []
    for file in os.listdir(path):
        files.append(file)
        # file_path = os.path.join(path, file)
        # if os.path.isdir(file_path):
        #     listdir(file_path)
    return files

# Cross-Validation
def crossVal(idx):
    # Construct testing Project files
    testProj = codePath + projDict[idx-1]
    test_X = []
    test_Y = []
    print("Start to read the Methods...")
    for f in os.listdir(testProj + 'allJavaFuncs/'):
        cur = os.path.join(testProj + 'allJavaFuncs/', f)
        with open(cur, 'r') as fx:
            test_X.append(fx.read())
    print("Start to read the Comments...")
    for f in os.listdir(testProj + 'allJavaComments/'):
        cur = os.path.join(testProj + 'allJavaComments/', f)
        with open(cur, 'r') as fy:
            test_Y.append(int(fy.read()[0]))

    # Construct training Project Files
    train_X = []
    train_Y = []
    trainProj = [x for x in proSet-{idx}]
    for tidx in trainProj:
        curp = codePath + projDict[tidx-1]
        print("Start to read {}'s Methods...".format(curp))
        for f in os.listdir(curp + 'allJavaFuncs/'):
            cur = os.path.join(curp + 'allJavaFuncs/', f)
            with open(cur, 'r') as fx:
                train_X.append(fx.read())
        print("Start to read {}'s Comments...".format(curp))
        for f in os.listdir(curp + 'allJavaComments/'):
            cur = os.path.join(curp + 'allJavaComments/', f)
            with open(cur, 'r') as fy:
                train_Y.append(int(fy.read()[0]))

    return [train_X, train_Y], [test_X, test_Y]

if __name__ == "__main__":
    # print(codePath + projDict[3])
    for i in range(10):
        trainData, testData = crossVal(i+1)
        trainX = trainData[0]
        trainY = trainData[0]
        testX = testData[0]
        testY = testData[1]
        print(len(trainY))
        break