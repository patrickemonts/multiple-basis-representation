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
        else:
            print(f"File {fname} does not exist. Skipping. ",file=sys.stderr)
            

    df = pd.concat(dfvec)
    if "degree" in df.columns:
        df["degree"] = df["degree"].fillna(-1)
        df["degree"] = df["degree"].astype(int)
        df["degree_str"] = df["degree"].apply(lambda x: f" - {x:d}" if x >0 else "")
    else:
        df["degree_str"] = ""
    df["type_plot"] = df["type"] + df["degree_str"]
    f, axvec = plt.subplots(3,1)
    sns.scatterplot(data=df,hue="type_plot",x="h",y="mz", style="type", alpha=0.9, ax=axvec[0])
    sns.scatterplot(data=df,hue="type_plot",x="h",y="mx", style="type", alpha=0.9, ax=axvec[1])
    sns.scatterplot(data=df,hue="type_plot",x="h",y="mx_s", style="type", alpha=0.9, ax=axvec[2])
    plt.show()

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("fnames", nargs="+", type=str, help="Filenames of the data files")

    args = parser.parse_args()
    main(args)