# Unsupervised LDA
Scripts to run experiments with LDA models. Run any script with the `-h` flag to see its arguments.

```
python <SCRIPT> -h
```

## Environment
Experiments require:
- `$DATA_DIR` environment variable pointing to a folder containing dataset files
- `$MODEL_DIR` environment variable pointing to a folder where models will be saved

## Run Experiments
- `ldaseq.py`: Batch-generate LDA models on a given corpus. Will output models into the directory structure described below. Requires a path to an experiment setup `.json` file. See below for the structure of this file.

## Explore Results
- `aggr_results.py`: Construct coherence plot for an experiment runs with dynamic LDA models.
- `top_words.py`: Given a specified number of topics and experiment `.json` file, load the topic keywords and coherence score for each timeslice. Will print all this information in Markdown-formatted text so that topics can be expanded using `<summary>`/`<details>` HTML tags. Optionally constructs a per-topic coherence plot.

## Experiment setup file
Run `ldaseq.py` with a path to a `.json` file structured as follows. All entries after `data_path` are optional.
```json
{
    "name": "unique experiment name",
    "min_topics": "int",
    "max_topics": "int",
    "days_in_interval": "int, the number of days per bucket for dynamic LDA",
    "text_key": "heading in the data table corresponding to the text of the posts",
    "data_path" : "path to data table",
    "attribute_filters": [
        {
            "filter_key": "data table heading",
            "filter_vals": [
                "list",
                "of",
                "values",
                "to",
                "include",
            ]
        }
    ],
    "time_filter": {
        "arg_format": "Python datetime formatting code for entries in this file",
        "start": "timestamp formatted according to arg_format",
        "end": "timestamp formatted according to arg_format",
        "data_format": "Python datetime formatting code for entries in the data (optional)",
        "time_key" : "heading in data table where timestamps appear"
    },
    "replace_before_stemming" : {
        "replace this": "with this",
        "and this": "with this"
    },
    "replace_after_stemming": {
        "same structure": "as above"
    },
    "remove_before_stemming": [
        "strings",
        "to",
        "remove"
    ],
    "remove_after_stemming": [
        "strings",
        "to",
        "remove"
    ],
    "passes": "int indicating how many passes to use in the initial LDA model"
}
```

## Model Output Structure
`ldaseq.py` trains a dynamic LDA model for each `n_topics` in [`min_topics`, `max_topics`]. Each model is evaluated for C_V coherence. The models are saved in a directory tree with the following structure. Each leaf directory contains a saved dynamic LDA model, coherence models for each timeslice, and a `metadata.json` file. The number of timeslices depends on `days_in_interval` and the overall timeslice which the data spans.

`metadata.json` files contain coherence scores for each model in that `n_topics`.

```bash
$MODEL_DIR
└───experiment_name
    ├───min_topics
    ├───min_topics+1
    │   ...
    └───max_topics
```
