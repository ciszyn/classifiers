import pandas as pd
import matplotlib.pyplot as plt

def plot(filename):
    df = pd.read_csv(filename)
    res = df['test_acc_2'].sub(df['test_acc_1'], axis=0).to_numpy()
    plt.figure()
    plt.hist(res)
    plt.title(filename.split('/')[-1])
    plt.savefig(filename[:-4]+'.png')

if __name__ == "__main__":
    method = 'differential_evolution'
    plot("./votingVars/resPlura"+method+".csv")
    plot("./votingVars/resKAppr"+method+".csv")
    plot("./votingVArs/resBorda"+method+".csv")