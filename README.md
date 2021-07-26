# Topic Modeling
Code for running robust and repeatable unsupervised topic modeling experiments. Use the `-h` flag to view CLI parameters for any script.

## Files and Folders
- [**unsupervised-lda**](./unsupervised-lda): Files related to the training and analysis of LDA topic models
- [**unsupervised-ldaseq**](./unsupervised-ldaseq): Files related to the training and analysis of dynamic topic models (using `gensim`'s `ldaseq` implementation)
- `list_common_words.py`: Takes an experiment config file as a command line argument and runs all specified preprocessing before listing the top 50 words in the dataset which will be used in that experiment
    - See [**unsupervised-lda**](./unsupervised-lda) or [**unsupervised-ldaseq**](./unsupervised-ldaseq) READMEs for the structure of an experiment JSON file
- `plot_data_quants.py`: Driver function to use a `TextParser` to make plots of the quantities of data in time frames (especially useful for deciding time intervals for a dynamic topic model)
