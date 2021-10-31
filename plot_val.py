import pickle
import argparse
import matplotlib.pyplot as plt

def plot(ip):
    with open(ip, "rb") as input_file:
        accus = pickle.load(input_file)

    for model in accus.keys():
        plt.figure()
        train_acc = []
        test_acc = []
        plt.xlabel("K-fold number")
        plt.ylabel("Accuracy")
        plt.title(model)
        plt.xlim(0,9)
        plt.ylim(0.65,1)
        for idx in range(10):
            train_acc.append(accus[model][idx][0])
            test_acc.append(accus[model][idx][1])
        plt.plot([i for i in range(0,10)],train_acc, c= "r",label = "train accuracy")
        plt.plot([i for i in range(0,10)],test_acc, c= "b",label = "test accuracy")
        plt.legend()
        plt.show()
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="kfold.py")
    parser.add_argument("-filename", default="data",help="Path to Data")
    args = parser.parse_args()
    plot(args.filename)
