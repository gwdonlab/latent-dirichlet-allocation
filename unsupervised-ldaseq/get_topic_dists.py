import os, json, datetime, csv
import argparse as ap
from ogm.trainer import TextTrainer
from ogm.utils import text_data_preprocess


def get_setup_dict():
    parser = ap.ArgumentParser()
    parser.add_argument("expt_config", help="Path to experiment's JSON file")
    parser.add_argument(
        "n_topics", help="Number of topics to identify saved Sequential LDA model", type=int
    )
    parser.add_argument("data_id", help="Column of dataset containing a unique ID value")
    parser.add_argument("--output_file", help="Name of CSV output file", default="output.csv")
    return parser.parse_args()


def main(args):
    with open(args.expt_config, "r") as infile:
        setup_dict = json.load(infile)

    # Load dataset and perform same preprocessing on it as the experiment
    if not os.path.isfile(setup_dict["data_path"]):
        setup_dict["data_path"] = os.getenv("DATA_DIR") + "/" + setup_dict["data_path"]
    preprocessed_data = text_data_preprocess(setup_dict, output=False)
    trainer = TextTrainer()
    trainer.data = preprocessed_data
    if "time_filter" not in setup_dict:
        raise ValueError("A time filter is required for training a sequential LDA")
    time_to_begin = datetime.datetime.strptime(
        setup_dict["time_filter"]["start"], setup_dict["time_filter"]["arg_format"]
    )
    if "end" in setup_dict["time_filter"]:
        time_to_end = datetime.datetime.strptime(
            setup_dict["time_filter"]["end"], setup_dict["time_filter"]["arg_format"]
        )
    else:
        time_to_end = datetime.datetime.now()

    trainer.filter_within_time_range(
        col=setup_dict["time_filter"]["time_key"],
        start=time_to_begin.strftime("%Y-%m-%d %H:%M:%S"),
        end=time_to_end.strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Load saved model
    model_savepath = (
        os.getenv("MODEL_DIR")
        + "/"
        + setup_dict["name"]
        + "/"
        + str(args.n_topics)
        + "topics/ldaseq.model"
    )
    trainer.load_model("ldaseq", model_savepath)

    # Model's gamma list had also better match the size of the dataset
    assert trainer.data.shape[0] == len(trainer.model.gammas)

    # Get topic distribution for each doc
    output_array = [
        [args.data_id] + ["topic_" + str(i) for i, _ in enumerate(trainer.model.doc_topics(0))]
    ]
    for doc_index in range(len(trainer.data)):
        doc_id = trainer[doc_index][args.data_id]
        topic_dist = trainer.model.doc_topics(doc_index)
        topic_dist = topic_dist.tolist()
        topic_dist.insert(0, doc_id)
        output_array.append(topic_dist)

    # Write distributions
    with open(args.output_file, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(output_array)


if __name__ == "__main__":
    main(get_setup_dict())
