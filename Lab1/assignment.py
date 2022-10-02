import json
from typing import Tuple, List
import numpy as np
from torch import argmax

def generate_uniform(seed: int, num_samples: int) -> None:
    """
    Generate 'num_samples' number of samples from uniform
    distribution and store it in 'uniform.txt'
    """

    # TODO  
    np.random.seed(seed=seed)
    x = np.random.uniform(0,1,size=num_samples)
    np.savetxt("uniform.txt", x, newline="\n")

    # END TODO

    assert len(np.loadtxt("uniform.txt", dtype=float)) == 100
    return None


def inv_transform(file_name: str, distribution: str, **kwargs) -> list:
    """ populate the 'samples' list from the desired distribution """
    samples = []

    # TODO
    uniform_samples = np.loadtxt(file_name, dtype=float)
    if(distribution == "categorical"):
        prob_dict = {}
        values = kwargs.get('values')
        probs = kwargs.get('probs')

        for key, value in zip(values, probs):
            prob_dict[key] = value
        
        sorted_prob_dict = dict(sorted(prob_dict.items()))
        cdf_dict = {}

        sum = 0.0
        for key, value in sorted_prob_dict.items():
            sum += value
            cdf_dict[key] = sum


        for i in range(uniform_samples.shape[0]):
            for key, value in cdf_dict.items():
                # print(uniform_samples[i])
                if(value > uniform_samples[i]):
                    samples.append(key)
                    break
        

        # np.savetxt("my_categorical.txt",samples)
    elif(distribution == "exponential"):
        samples = -np.log(1-uniform_samples) / kwargs.get('lambda')
        # np.savetxt("my_exponential.txt",samples)
    elif(distribution == "cauchy"):
        samples = np.tan(np.pi * (uniform_samples - 0.5)) * kwargs.get('gamma') + kwargs.get('peak_x')
        # np.savetxt("my_cauchy.txt",samples)

    # END TODO
    assert len(samples) == 100
    return samples


def find_best_distribution(samples: list) -> Tuple[int, int, int]:
    """
    Given the three distributions of three different types, find the distribution
    which is most likely the data is sampled from for each type
    Return a tupple of three indices corresponding to the best distribution
    of each type as mentioned in the problem statement
    """
    indices = [0,0,0]

    def pdf_gaussian(x, mu: float, sigma: float):
        return np.log(np.exp(-0.5*((x-mu)/sigma)**2)/(np.sqrt(2*np.pi)*sigma))

    def pdf_expo(x, lmbd: float):
        return np.log(lmbd * np.exp(-lmbd*x))

    def pdf_uniform(x, a, b):
        return [np.log(1/(b-a)) if (i>=a and i<=b) else 0 for i in x]

    # TODO
    #gaussian
    mle_0 = np.sum(pdf_gaussian(samples, mu=0,sigma= 1))
    mle_1 = np.sum(pdf_gaussian(samples, mu=0,sigma= 0.5))
    mle_2 = np.sum(pdf_gaussian(samples, mu=1,sigma= 1))

    indices[0] = np.argmax(np.array([mle_0,mle_1,mle_2]))
    #uniform   
    mle_0 = np.sum(pdf_uniform(samples, 0,1))
    mle_1 = np.sum(pdf_uniform(samples, 0,2))
    mle_2 = np.sum(pdf_uniform(samples, -1,1))

    # print(np.argmax([1,2,3]))

    indices[1] = np.argmax(np.array([mle_0,mle_1,mle_2]))

    #Exponential
    mle_0 = np.sum(pdf_expo(samples, lmbd =  0.5))
    mle_1 = np.sum(pdf_expo(samples, lmbd = 1))
    mle_2 = np.sum(pdf_expo(samples, lmbd = 2))

    indices[2] = np.argmax(np.array([mle_0,mle_1,mle_2]))

    indices = tuple(indices)
    # END TODO
    assert len(indices) == 3
    assert all([index >= 0 and index <= 2 for index in indices])
    return indices

def marks_confidence_intervals(samples: list, variance: float, epsilons: list) -> Tuple[float, List[float]]:

    sample_mean = 0
    deltas = [0 for e in epsilons] # List of zeros

    # TODO
    sample_mean = np.mean(samples)
    n = len(samples)
    deltas = [variance/(n*e*e) for e in epsilons]

    # END TODO

    assert len(deltas) == len(epsilons)
    return sample_mean, deltas

if __name__ == "__main__":
    seed = 21734

    # question 1
    generate_uniform(seed, 100)

    # question 2
    for distribution in ["categorical", "exponential", "cauchy"]:
        file_name = "q2_" + distribution + ".json"
        args = json.load(open(file_name, "r"))
        samples = inv_transform(**args)
        with open("q2_output_" + distribution + ".txt", "w") as f:
            for elem in samples:
                f.write(str(elem) + "\n")

    # question 3
    indices = find_best_distribution(np.loadtxt("q3_samples.csv", dtype=float))
    with open("q3_output.txt", "w") as f:
        f.write("\n".join([str(e) for e in indices]))

    # question 4
    q4_samples = np.loadtxt("q4_samples.csv", dtype=float)
    q4_epsilons = np.loadtxt("q4_epsilons.csv", dtype=float)
    variance = 5

    sample_mean, deltas = marks_confidence_intervals(q4_samples, variance, q4_epsilons)

    with open("q4_output.txt", "w") as f:
        f.write("\n".join([str(e) for e in [sample_mean, *deltas]]))
