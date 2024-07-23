import pandas as pd
import numpy as np
import os, sys
import seaborn as sns
import matplotlib.pyplot as plt

def main(args):
    dfvec = []
    for fname in args.fnames:
        if os.path.exists(fname):
            dfvec.append(pd.read_csv(fname, index_col=0))

    df = pd.concat(dfvec)
    df["degree_str"] = df["degree"].apply(lambda x: " - "+str(x) if x is not np.nan else "")
    df["type_plot"] = df["type"] + df["degree_str"]
    sns.relplot(data=df,hue="type_plot",x="h",y="energy", col="nx", style="type", alpha=0.9)
    plt.show()

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("fnames", nargs="+", type=str, help="Filenames of the data files")

    args = parser.parse_args()
    main(args)