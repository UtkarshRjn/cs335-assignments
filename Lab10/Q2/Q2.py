import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", type=str, required=True, help="Path to the training json file"
    )

    parser.add_argument(
        "--input", type=str, required=True, help="Path to the test input json file"
    )

    parser.add_argument(
        "--output", type=str, required=True, help="Path to the test output json file"
    )

    # parser.add_argument(
    #     "--prediction", type=str, required=True, help="Path to the actual lable files"
    # )

    x = []
    y = []

    args = parser.parse_args()

    sub_to_int = {
        "Physics":0, "Chemistry":1 ,"PhysicalEducation":2,"English":3,"Economics":4,"Biology":5,"ComputerScience":6,"Accountancy":7,"BusinessStudies":8   
    }
    with open(args.train, 'r') as f:
        i = 0
        for line in f:
            line = line.strip().split()[0]
            if i != 0:
                scores = json.loads(line)
                vec = np.zeros((9,))
                j = 0
                for sub, grade in scores.items():
                    if(j <= 3): vec[sub_to_int[sub]] = grade

                    if(j == 4): y.append(grade)
                    j += 1

                x.append(vec)
            i += 1

    x = np.array(x)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)
    clf = RandomForestClassifier(max_depth=10)
    clf.fit(x, y)

    x_test = []

    args = parser.parse_args()

    with open(args.input, 'r') as f:
        i = 0
        for line in f:
            line = line.strip().split()[0]
            if i != 0:
                scores = json.loads(line)
                vec = np.zeros((9,))
                j = 0
                for sub, grade in scores.items():
                    if(j <= 3): vec[sub_to_int[sub]] = grade
                    j += 1

                x_test.append(vec)
            i += 1

    y_test = clf.predict(x_test)

    # ans = np.zeros((y_test.shape[0],))
    # with open(args.prediction, 'r') as f:
    #     i = 0
    #     for line in f:
    #         line = line.strip().split()[0]
    #         ans[i] = int(line)
    #         i += 1

    # correct = 0
    # incorrect = 0
    # for i in range(y_test.shape[0]):
    #     if(abs(y_test[i] - ans[i]) <= 1): correct += 1
    #     else: incorrect += 1

    # print("Correct: ", correct)
    # print("Incorrect: ", incorrect)
    # print("accuracy: ", correct/(correct+incorrect))
    # print("score:", (correct-incorrect)/(correct+incorrect))

    np.savetxt(args.output,y_test.astype(int),fmt='%i')