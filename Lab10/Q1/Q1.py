import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input file"
    )

    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output file"
    )

    
    args = parser.parse_args()
    
    missing_indices = []
    y = []
    x = []
    
    with open(args.input) as f:
        i = 0
        for line in f:
            line = line.strip().split()
            if i==0:
                numTestCases = int(line[0])
            else:
                try:
                    y.append(float(line[2]))
                    x.append(i-1)
                except:
                    missing_indices.append(i-1)
            
            i += 1


    missing_indices = np.array(missing_indices)
    print(missing_indices)
    y = np.array(y)
    x = np.array(x)

    x = x.reshape(x.shape[0],1)
    y = y.reshape(y.shape[0],1)
    missing_indices = missing_indices.reshape(missing_indices.shape[0],1)

    poly = PolynomialFeatures(degree = 8)
    X_poly = poly.fit_transform(x)
    poly.fit(X_poly, y)
    lin2 = LinearRegression()
    lin2.fit(X_poly, y)

    plt.scatter(x, y, color = 'blue')
    plt.plot(x, lin2.predict(poly.fit_transform(x)), color = 'red')
    plt.show()
    y_pred = lin2.predict(poly.fit_transform(missing_indices))

    np.savetxt(args.output,y_pred.astype(float),fmt='%.2f')


