import os, json
import pandas as pd
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
    parser.add_argument("--output_file", help="Name of Excel output file", default="output.xlsx")
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
        raise ValueError("A time filter is required for running a sequential LDA")
    time_to_begin = pd.Timestamp(setup_dict["time_filter"]["start"])
    time_to_end = (
        pd.Timestamp(setup_dict["time_filter"]["end"])
        if "end" in setup_dict["time_filter"]
        else pd.Timestamp.now()
    )

    trainer.filter_within_time_range(
        col=setup_dict["time_filter"]["time_key"],
        start=time_to_begin.strftime("%Y-%m-%d %H:%M:%S"),
        end=time_to_end.strftime("%Y-%m-%d %H:%M:%S"),
        input_format="%Y-%m-%d %H:%M:%S",
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
    output_array = []
    for doc_index in range(len(trainer.data)):
        doc_id = trainer[doc_index][args.data_id]
        topic_dist = trainer.model.doc_topics(doc_index)
        topic_dist = topic_dist.tolist()
        topic_dist.insert(0, doc_id)
        output_array.append(topic_dist)

    # Write distributions
    pd.DataFrame(
        output_array,
        columns=[args.data_id]
        + ["topic_" + str(i) for i, _ in enumerate(trainer.model.doc_topics(0))],
    ).to_excel(args.output_file, index=False)


if __name__ == "__main__":
    main(get_setup_dict())
