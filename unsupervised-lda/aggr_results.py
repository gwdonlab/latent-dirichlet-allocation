import json, os
import numpy as np
import argparse as ap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# Experiment parameters obtained by CLI args
argparser = ap.ArgumentParser()
argparser.add_argument("plot_title", help="Title to appear on plot")
argparser.add_argument(
    "experiment_configs",
    help="1 or more paths to experiment config JSON files",
    nargs="+",
)
argparser.add_argument(
    "--y_title", default="Experiment Names", help="Title for the y-axis in a 3D plot"
)
argparser.add_argument(
    "--plot_20news",
    required=False,
    dest="plot_20news",
    help="If set, will add a coherence line for the 20 Newsgroups test dataset",
    action="store_true",
)
argparser.add_argument(
    "--plot_3d",
    required=False,
    help="If set, will attempt to generate a 3D plot of coherence, experiments, and number of topics",
    dest="plot_3d",
    action="store_true",
)
argparser.add_argument(
    "--no_errorbars",
    required=False,
    help="If set, removes error bars on plots",
    dest="no_errorbars",
    action="store_true",
)
argparser.add_argument(
    "--legend",
    required=False,
    help="If set, adds a plot legend; the keys in this legend will be experiment names unless 'plot_name' is in the config file",
    dest="add_legend",
    action="store_true",
)
args = argparser.parse_args()

# Preliminary error-checking
if args.plot_3d and args.plot_20news:
    print("Plotting 20 News as a 3D plot is currently unsupported")
    args.plot_20news = False
if args.plot_3d:
    y_axis_labels = [x[:3] + "..." + x[-3:] for x in args.experiment_names]

if args.plot_3d:
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # A 2D array of the topics for all the experiments (should be the same for all experiments)
    all_topics = []

    # A 2D array of all the coherence scores for the experiments
    all_coherences = []
    for experiment_path in args.experiment_configs:
        # Load experiment config file
        with open(experiment_path, "r") as infile:
            expt_config = json.load(infile)
        experiment_name = expt_config["name"]

        # Read in all metadata files
        models_parent_dir = os.getenv("MODEL_DIR") + "/" + experiment_name
        topic_num_subdirs = os.listdir(models_parent_dir)
        json_files = [
            models_parent_dir + "/" + x + "/metadata.json" for x in topic_num_subdirs
        ]

        x_topics = []
        y_coherence = []

        for filename in json_files:
            try:
                with open(filename, "r") as json_file:
                    info = json.load(json_file)
                    x_topics.append(info["aggregated"]["topics"])
                    y_coherence.append(info["aggregated"]["avg_coherence"])
            except FileNotFoundError:
                print("Couldn't find " + filename)

        temp = zip(x_topics, y_coherence)
        res = sorted(temp, key=lambda x: x[0])
        x_topics, z_coherence = zip(*res)

        all_topics.append(x_topics)
        all_coherences.append(z_coherence)

    # Check to ensure the all_topics array is correct
    x_lengths = [len(x) for x in all_topics]
    if len(set(x_lengths)) > 1:
        raise ValueError("Experiments have different numbers of topics")

    # Map the experiment names to numeric data for the y-axis
    numerical_labels = np.arange(len(args.experiment_names))
    X, Y = np.meshgrid(all_topics[0], numerical_labels)

    # Ensure x-axis is labeled with integers
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    surf = ax.plot_surface(
        X, Y, np.array(all_coherences), cmap=cm.coolwarm, linewidth=0, antialiased=False
    )

    # Set up the color bar
    cbaxes = fig.add_axes([0.05, 0.1, 0.03, 0.5])
    cb = plt.colorbar(surf, cax=cbaxes)

    # Label y-axis with the specified labels
    ax.set_yticks(numerical_labels)
    ax.set_yticklabels(y_axis_labels)

    # Label the y-axis
    if args.y_title is not None:
        ax.set_ylabel(args.y_title)


else:
    ax = plt.figure().gca()

    for experiment_path in args.experiment_configs:
        # Load experiment config file
        with open(experiment_path, "r") as infile:
            expt_config = json.load(infile)
        experiment_name = expt_config["name"]

        # Read in all metadata files
        models_parent_dir = os.getenv("MODEL_DIR") + "/" + experiment_name
        topic_num_subdirs = os.listdir(models_parent_dir)
        json_files = [
            models_parent_dir + "/" + x + "/metadata.json" for x in topic_num_subdirs
        ]

        x_topics = []
        y_coherence = []
        y_err = []

        for filename in json_files:
            try:
                with open(filename, "r") as json_file:
                    info = json.load(json_file)
                    x_topics.append(info["aggregated"]["topics"])
                    y_coherence.append(info["aggregated"]["avg_coherence"])
                    y_err.append(info["aggregated"]["coherence_stdev"])
            except FileNotFoundError:
                print("Couldn't find " + filename)

        temp = zip(x_topics, y_coherence, y_err)
        res = sorted(temp, key=lambda x: x[0])
        x_topics, y_coherence, y_err = zip(*res)

        # Plot topic_num vs coherence
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        if args.no_errorbars:
            if "plot_name" in expt_config:
                ax.plot(x_topics, y_coherence, label=expt_config["plot_name"])
            else:
                ax.plot(x_topics, y_coherence, label=experiment_name)

        else:
            if "plot_name" in expt_config:
                ax.errorbar(
                    x_topics, y_coherence, yerr=y_err, label=expt_config["plot_name"]
                )
            else:
                ax.errorbar(x_topics, y_coherence, yerr=y_err, label=experiment_name)

    if args.plot_20news:
        with open(
            os.getenv("MODEL_DIR") + "/20Newsgroups/metadata.json", "r"
        ) as baseline_info:
            baseline_dict = json.load(baseline_info)
            baseline = [
                baseline_dict["aggregated"]["avg_coherence"]
                for i in range(len(x_topics))
            ]

        ax.plot(x_topics, baseline, "k--", label="20News baseline")

if args.plot_3d:
    ax.set_zlabel("Coherence score ($C_v$)")
else:
    ax.set_ylabel("Coherence score ($C_v$)")

    if args.add_legend:
        ax.legend()

ax.set_xlabel("Number of topics")
ax.set_title(args.plot_title)
plt.show()
