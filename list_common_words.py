from ogm.parser import TextParser
from collections import Counter
import argparse as ap
import os, json


def get_setup_dict():
    p = ap.ArgumentParser()
    p.add_argument("filepath", help="Path to experiment's JSON file")
    a = p.parse_args()
    with open(a.filepath, "r") as infile:
        input_dict = json.load(infile)

    return input_dict


def main(setup_dict):

    # Path to data file
    dataf = os.getenv("DATA_DIR")
    data_file = dataf + "/" + setup_dict["data_path"]

    # Read in data and run the gensim preprocessing on it
    trainer = TextParser()
    trainer.parse_file(data_file)

    if "time_filter" in setup_dict:
        trainer.filter_within_time_range(
            setup_dict["time_filter"]["time_key"],
            setup_dict["time_filter"]["data_format"],
            setup_dict["time_filter"]["arg_format"],
            setup_dict["time_filter"]["start"],
            setup_dict["time_filter"]["end"],
        )

    if "attribute_filters" in setup_dict:
        for attr_filter in setup_dict["attribute_filters"]:
            trainer.filter_data(
                attr_filter["filter_key"], set(attr_filter["filter_vals"])
            )

    print("Found " + str(len(trainer.data)) + " posts")

    if "replace_before_stemming" in setup_dict:
        trainer.replace_words(
            setup_dict["text_key"], setup_dict["replace_before_stemming"]
        )

    if "remove_before_stemming" in setup_dict:
        trainer.remove_words(
            setup_dict["text_key"], set(setup_dict["remove_before_stemming"])
        )

    trainer.lemmatize_stem_words(setup_dict["text_key"])

    if "replace_after_stemming" in setup_dict:
        trainer.replace_words(
            setup_dict["text_key"], setup_dict["replace_after_stemming"]
        )

    if "remove_after_stemming" in setup_dict:
        trainer.remove_words(
            setup_dict["text_key"], set(setup_dict["remove_after_stemming"])
        )

    text_key = setup_dict["text_key"]

    all_texts_concat = []
    for x in trainer.data:
        for word in x[text_key]:
            all_texts_concat.append(word)

    c = Counter(all_texts_concat)
    print(c.most_common(50))


if __name__ == "__main__":
    d = get_setup_dict()
    main(d)
