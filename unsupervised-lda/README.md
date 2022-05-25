# Unsupervised LDA
Scripts to run experiments with LDA models. Run any script with the `-h` flag to see its arguments.

```
python <SCRIPT> -h
```

## Environment
LDA experiments require:
- `$DATA_DIR` environment variable pointing to a folder containing dataset files
- `$MODEL_DIR` environment variable pointing to a folder where models will be saved

## Run Experiments
- `lda.py`: Batch-generate LDA models on a given corpus. Will output models into the directory structure described below. Requires a path to an experiment setup `.json` file. See below for the structure of this file.
- `generate_20news_baseline.py`: Train 10 LDA models with 20 topics over the 20 Newsgroups dataset included with Scikit. This should be a decent baseline for coherence scores.

## Explore Results
- `aggr_results.py`: Construct coherence plot for one or many experiment runs with LDA models.
    - **Note:** 3-D plotting is not compatible with additional coherence scores. Only *C_V* will be plotted.
- `calculate_coherence.py`: Calculate alternate coherence scores than just *C_V*
- `top_words.py`: Load the model with the best coherence score (given a specified number of topics and experiment `.json` file which generated the model) and output the probability distribution for words in its topics. Will also output a per-topic coherence score. This script also has some additional dependencies for optional features that are set to `False` by default.
    - To save an LDAvis HTML file for better visualization, you need the `pyLDAvis` package
    - To generate a word cloud, you need the `Pillow` and `wordcloud` packages.
- `convert_plot_dumped_axes.py`: Transpose dumped word cloud data from `top_words.py` into CSV files and, optionally, Matplot graphs. Some people like bar plots better than word clouds.

## Experiment setup file
Run `lda.py` with a path to a `.json` file structured as follows. All entries after `data_path` are optional.
```json
{
    "name": "unique experiment name",
    "min_topics": "int",
    "max_topics": "int",
    "n_trials": "int, the number of models to train per n_topics",
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
        "data_format": "Python datetime formatting code for entries in the data",
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
    "plot_name": "human-readable experiment name to put in a plot legend",
    "lda_nosave": "boolean; if true, will suppress saving of LDA models",
    "coherence_nosave": "boolean; if true, will suppress saving of coherence models"
}
```

## Model Output Structure
`lda.py` trains `n_trials` LDA models for each `n_topics` in [`min_topics`, `max_topics`]. Each model is evaluated for C_V coherence. The models are saved in a directory tree with the following structure. Each leaf directory contains a saved LDA model and its corresponding coherence model, `gensim` dictionary, expElogbeta `numpy` array, and model state.

`metadata.json` files contain coherence scores for each model in that `n_topics`.

```bash
$MODEL_DIR
└───experiment_name
    ├───min_topics
    │   ├───model_0
    │   ├───model_1
    │   │   ...
    │   ├───n_trials
    │   └───metadata.json
    ├───min_topics+1
    │   ├───model_0
    │   ├───model_1
    │   │   ...
    │   ├───n_trials
    │   └───metadata.json
    │   ...
    └───max_topics
        ├───model_0
        ├───model_1
        │   ...
        ├───n_trials
        └───metadata.json
```
