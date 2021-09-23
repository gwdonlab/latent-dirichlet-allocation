import os, json, datetime
import argparse as ap
import numpy as np
from ogm.trainer import TextTrainer
from ogm.utils import text_data_preprocess
from gensim.models import CoherenceModel


def get_setup_dict():
    p = ap.ArgumentParser()
    p.add_argument("filepath", help="Path to experiment's JSON file")
    p.add_argument(
        "--show_data_plot",
        action="store_true",
        help="If set, will plot a histogram of the data quanities over time",
    )
    a = p.parse_args()
    with open(a.filepath, "r") as infile:
        input_dict = json.load(infile)
    input_dict["show_plot"] = a.show_data_plot

    return input_dict


def main(setup_dict):

    # Look for input file at path and DATA_DIR if it's not there
    if not os.path.isfile(setup_dict["data_path"]):
        setup_dict["data_path"] = os.getenv("DATA_DIR") + "/" + setup_dict["data_path"]

    # Read in data and run the ogm preprocessing on it
    preprocessed_data = text_data_preprocess(setup_dict, output=False)

    # Add to Trainer object
    trainer = TextTrainer(
        log=setup_dict["name"] + str(setup_dict["min_topics"]) + ".log"
    )
    trainer.data = preprocessed_data

    if "time_filter" not in setup_dict:
        raise ValueError("A time filter is required for training a sequential LDA")

    print("Found " + str(len(trainer.data)) + " posts")

    time_to_begin = datetime.datetime.strptime(
        setup_dict["time_filter"]["start"], setup_dict["time_filter"]["arg_format"]
    )

    if "end" in setup_dict["time_filter"]:
        time_to_end = datetime.datetime.strptime(
            setup_dict["time_filter"]["end"], setup_dict["time_filter"]["arg_format"]
        )
    else:
        time_to_end = datetime.datetime.now()

    if "data_format" in setup_dict["time_filter"]:
        buckets, quants = trainer.plot_data_quantities(
            key=setup_dict["time_filter"]["time_key"],
            data_format=setup_dict["time_filter"]["data_format"],
            days_interval=setup_dict["days_in_interval"],
            start_date=time_to_begin.strftime(setup_dict["time_filter"]["data_format"]),
            end_date=time_to_end.strftime(setup_dict["time_filter"]["data_format"]),
            show_plot=setup_dict["show_plot"],
        )
    else:
        trainer.add_datetime_attribute(
            setup_dict["time_filter"]["time_key"], "__added_datetime"
        )
        buckets, quants = trainer.plot_data_quantities(
            key=setup_dict["time_filter"]["time_key"],
            days_interval=setup_dict["days_in_interval"],
            start_date=time_to_begin.strftime(setup_dict["time_filter"]["arg_format"]),
            end_date=time_to_end.strftime(setup_dict["time_filter"]["arg_format"]),
            data_format=setup_dict["time_filter"]["arg_format"],
            show_plot=setup_dict["show_plot"],
        )

    # Sum of bucket quantities had better match the size of the dataset
    assert sum(quants) == len(trainer.data)
    print("Training models with sequences:", buckets, quants)

    topic_quants = range(setup_dict["min_topics"], setup_dict["max_topics"] + 1)
    text_key = setup_dict["text_key"]
    experiment_name = setup_dict["name"]

    print("Training models for topic_nums:", topic_quants)

    # Loop through different topic quantities
    for num_topics in topic_quants:
        metadata = {}
        coherences = []

        # Create directory where model files will be saved
        model_savepath = (
            os.getenv("MODEL_DIR")
            + "/"
            + experiment_name
            + "/"
            + str(num_topics)
            + "topics"
        )
        os.makedirs(model_savepath, exist_ok=True)

        # Train model
        trainer.train_ldaseq(
            key=text_key,
            n_topics=num_topics,
            output_path=model_savepath + "/ldaseq.model",
            seq_counts=quants,
        )

        # Loop through the different time slices and get coherence at each one
        for i in range(len(quants)):
            model_output = trainer.model.dtm_coherence(i)
            cm = CoherenceModel(
                corpus=trainer.corpus,
                texts=trainer.get_attribute_list(text_key),
                topics=model_output,
                coherence="c_v",
                dictionary=trainer.dictionary,
            )

            # Save information about this time slice
            coherence = cm.get_coherence()
            coherences.append(coherence)
            metadata["time_" + str(i)] = {
                "coherence": coherence,
                "start_time": buckets[i],
                "num_posts": quants[i],
                "coherence_savepath": model_savepath
                + "/coherence_"
                + str(i)
                + ".model",
            }

            # Save coherence model
            cm.save(model_savepath + "/coherence_" + str(i) + ".model")

        # Save information about the coherence scores over all the time slices
        coherences = np.array(coherences)
        metadata["aggregated"] = {
            "avg_coherence": np.mean(coherences),
            "coherence_stdev": np.std(coherences),
            "coherence_variance": np.var(coherences),
            "topics": num_topics,
        }

        with open(
            os.getenv("MODEL_DIR")
            + "/"
            + experiment_name
            + "/"
            + str(num_topics)
            + "topics/metadata.json",
            "w",
        ) as output:
            json.dump(metadata, output)


if __name__ == "__main__":
    d = get_setup_dict()
    main(d)
