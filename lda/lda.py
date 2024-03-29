import os, json
import argparse as ap
import numpy as np
from ogm.trainer import TextTrainer
from gensim.models import CoherenceModel


def get_setup_dict():
    p = ap.ArgumentParser()
    p.add_argument("filepath", help="Path to experiment's JSON file")
    a = p.parse_args()
    with open(a.filepath, "r") as infile:
        input_dict = json.load(infile)

    return input_dict


def main(setup_dict):

    # Look for input file at path and DATA_DIR if it's not there
    if not os.path.isfile(setup_dict["data_path"]):
        data_file = os.getenv("DATA_DIR") + "/" + setup_dict["data_path"]
    else:
        data_file = setup_dict["data_path"]

    # Read in data and run the gensim preprocessing on it
    trainer = TextTrainer()
    trainer.parse_file(data_file)

    if "time_filter" in setup_dict:
        trainer.filter_within_time_range(
            col=setup_dict["time_filter"]["time_key"],
            data_format=setup_dict["time_filter"]["data_format"],
            input_format=setup_dict["time_filter"]["arg_format"],
            start=setup_dict["time_filter"]["start"],
            end=setup_dict["time_filter"]["end"],
        )

    if "attribute_filters" in setup_dict:
        for attr_filter in setup_dict["attribute_filters"]:
            trainer.filter_data(attr_filter["filter_key"], set(attr_filter["filter_vals"]))

    print("Found " + str(trainer.data.shape[0]) + " posts")

    if "replace_before_stemming" in setup_dict:
        trainer.replace_words(setup_dict["text_key"], setup_dict["replace_before_stemming"])

    if "remove_before_stemming" in setup_dict:
        a = trainer.remove_words(setup_dict["text_key"], set(setup_dict["remove_before_stemming"]))
        print("Removed " + str(a) + " instances of", setup_dict["remove_before_stemming"])

    trainer.lemmatize_stem_words(setup_dict["text_key"])

    if "replace_after_stemming" in setup_dict:
        trainer.replace_words(setup_dict["text_key"], setup_dict["replace_after_stemming"])

    if "remove_after_stemming" in setup_dict:
        a = trainer.remove_words(setup_dict["text_key"], set(setup_dict["remove_after_stemming"]))
        print("Removed " + str(a) + " instances of", setup_dict["remove_after_stemming"])

    topic_quants = range(setup_dict["min_topics"], setup_dict["max_topics"] + 1)
    text_key = setup_dict["text_key"]
    n_trials = setup_dict["n_trials"]
    experiment_name = setup_dict["name"]

    print("Training models for topic_nums:", topic_quants)

    # Loop through different topic quantities
    for num_topics in topic_quants:

        # For each topic quantity, run n_trials experiments
        metadata = {}
        coherences = []
        for i in range(n_trials):
            model_savepath = (
                os.getenv("MODEL_DIR")
                + "/"
                + experiment_name
                + "/"
                + str(num_topics)
                + "topics/model_"
                + str(i)
            )
            os.makedirs(model_savepath, exist_ok=True)

            # Check whether to save LDA model to disk
            if "lda_nosave" in setup_dict and setup_dict["lda_nosave"]:
                lda_savepath = None
            else:
                lda_savepath = model_savepath + "/lda.model"

            # Check whether to save coherence model to disk
            if "coherence_nosave" in setup_dict and setup_dict["coherence_nosave"]:
                c_savepath = None
            else:
                c_savepath = model_savepath + "/coherence.model"

            # Train a parallelized LDA model
            # ALPHA: has to do with the expected number of topics per document;
            # can be set to a `num_topics` length array representing each topic's probability,
            # or just a uniform distribution by default
            # BETA (eta in this implementation): has to do with the number of words per topic;
            # high beta means each topic has a mixture of most words,
            # low beta means each topic has a mixture of just a few of the words
            trainer.train_lda(
                col=text_key, n_topics=num_topics, output_path=lda_savepath, n_workers=8
            )

            print(
                "["
                + str(i + 1)
                + "/"
                + str(n_trials)
                + "]["
                + str(num_topics)
                + " topics] Model complete!"
            )

            # Make a coherence model for this LDA model
            cm = CoherenceModel(
                model=trainer.model,
                corpus=trainer.corpus,
                texts=trainer.get_attribute_list(text_key),
                coherence="c_v",
            )

            # Save information about this model
            coherence = cm.get_coherence()
            coherences.append(coherence)
            metadata["model_" + str(i)] = {
                "path": model_savepath,
                "coherence": coherence,
            }

            # Save coherence model
            if c_savepath:
                cm.save(c_savepath)

            print(
                "["
                + str(i + 1)
                + "/"
                + str(n_trials)
                + "]["
                + str(num_topics)
                + " topics] Coherence complete!"
            )

        # Save information about the coherence scores overall
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
