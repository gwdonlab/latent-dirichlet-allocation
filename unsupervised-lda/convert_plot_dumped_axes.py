import pickle, os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import random
import argparse as ap

argparser = ap.ArgumentParser()
argparser.add_argument(
    "--create_plots",
    required=False,
    help="If set, will attempt to plot topic keywords",
    action="store_true",
)
args = argparser.parse_args()

dataframes = [x for x in os.listdir() if x.endswith(".pkl")]
random.seed(45)

for f in os.listdir():
    if f.endswith(".pkl"):
        with open(f, "rb") as infile:
            axes = pickle.load(infile)
            if len(axes) != 2:
                continue

        x_axis = axes[0]
        y_axis = axes[1]

        df = pd.DataFrame({"Word": x_axis, "Weight": y_axis})
        df["Weight"] = df["Weight"].astype(float)

        dataframes[int(f.split("_")[1][:-4])] = df

if args.create_plots:
    fig, axs = plt.subplots(len(dataframes))
    fig.set_tight_layout(tight=True)
    fig.set_size_inches(7, 15)

for i, df in enumerate(dataframes):
    if args.create_plots:
        axs[i].bar(
            list(df["Word"]),
            list(df["Weight"]),
            color=(random.random(), random.random(), random.random()),
        )
        axs[i].set_title("Topic " + str(i + 1))
        axs[i].set_ylabel("Weight")
        axs[i].tick_params(axis="x", rotation=45)
        start, end = axs[i].get_ylim()
        axs[i].yaxis.set_ticks(np.arange(start, end, 0.005))
        axs[i].yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.3f"))
        axs[i].grid()

    df.to_csv("Topic" + str(i + 1) + ".csv", index=False)

if args.create_plots:
    plt.savefig("topics.svg")
    plt.savefig("topics.png")
