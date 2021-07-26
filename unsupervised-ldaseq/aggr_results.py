import json, os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import argparse as ap

# Experiment parameters obtained by CLI args
argparser = ap.ArgumentParser()
argparser.add_argument("--plot_title", help="Title to appear on plot")
argparser.add_argument(
    "--experiment_name",
    help="Experiment name corresponding to one specified in an experiment JSON file",
)
argparser.add_argument(
    "--legend",
    required=False,
    help="If set, adds a plot legend",
    dest="add_legend",
    action="store_true",
)
argparser.add_argument(
    "--remove_from_label",
    nargs="*",
    help="List of strings to remove from x-axis plot labels",
)
argparser.add_argument(
    "--label_rotation", type=int, default=90, help="Degrees to rotate the x-axis labels"
)
args = argparser.parse_args()


# Read in all metadata files
models_parent_dir = os.getenv("MODEL_DIR") + "/" + args.experiment_name
topic_num_subdirs = os.listdir(models_parent_dir)
json_files = [models_parent_dir + "/" + x + "/metadata.json" for x in topic_num_subdirs]
topic_nums = [int(x.replace("topics", "")) for x in topic_num_subdirs]

ax = plt.figure().gca()

# This will loop through all the "metadata" files, corresponding to different n_topics
for filename, n_topics in zip(json_files, topic_nums):
    x_time = []
    y_coherence = []

    try:
        with open(filename, "r") as json_file:
            info = json.load(json_file)

            # This is how many timeslices there were
            # Each timeslice has its own section, but there's
            # one extra section listed as "aggregated"
            for i in range(len(info.keys()) - 1):
                x_label = info["time_" + str(i)]["start_time"]
                if args.remove_from_label is not None:
                    for item in args.remove_from_label:
                        x_label = x_label.replace(item, "")

                x_time.append(x_label)
                y_coherence.append(info["time_" + str(i)]["coherence"])

        # Plot time vs coherence for each n_topics
        plt.xticks(rotation=args.label_rotation)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.plot(x_time, y_coherence, label=str(n_topics) + " Topics")

    # Since LdaSeq takes so long to train, it may serve to analyze before it finishes
    except FileNotFoundError:
        continue


ax.set_ylabel("Coherence score ($C_v$)")

if args.add_legend:
    ax.legend()

ax.set_xlabel("Start of Timeslice")
ax.set_title(args.plot_title)
plt.show()
