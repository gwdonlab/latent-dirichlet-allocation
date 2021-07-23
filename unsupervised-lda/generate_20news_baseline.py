import os
import gensim
import json
from ogm.parser import TextParser
from sklearn.datasets import fetch_20newsgroups
import numpy as np


def main():
    # Load newsgroup data
    newsgroup_data = fetch_20newsgroups(subset="all")

    parser = TextParser()
    parser.data = []
    for article in newsgroup_data.data:
        item = {}
        item["text"] = article
        parser.data.append(item)
    parser.lemmatize_stem_words("text")
    processed_docs = [x["text"] for x in parser.data]

    # Essentially a gensim-flavored dict mapping words to their BoW ID
    dictionary = gensim.corpora.Dictionary(processed_docs)

    # bow_corpus is a list (length of dataset size, each entry represents a document) of tuples (token_id, token_count)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # Run and average 10 different models
    metadata = {}
    coherences = []
    for i in range(10):

        print("Training LDA model " + str(i))

        # Make LDA model
        lda_model = gensim.models.LdaMulticore(
            bow_corpus, num_topics=20, id2word=dictionary, passes=10, workers=4
        )

        # Write LDA model to disk
        model_savepath = os.getenv("MODEL_DIR") + "/20Newsgroups/model_" + str(i)
        os.mkdir(model_savepath)
        lda_model.save(model_savepath + "/lda.model")
        print("Model saved. Getting coherence.")

        # Get coherence for model
        cm = gensim.models.CoherenceModel(
            model=lda_model, corpus=bow_corpus, texts=processed_docs, coherence="c_v"
        )

        # Save information about this model
        coherence = cm.get_coherence()
        coherences.append(coherence)
        metadata["model_" + str(i)] = {"path": model_savepath, "coherence": coherence}

    # Save information about the coherence scores overall
    coherences = np.array(coherences)
    metadata["aggregated"] = {
        "avg_coherence": np.mean(coherences),
        "coherence_stdev": np.std(coherences),
        "coherence_variance": np.var(coherences),
    }

    with open(os.getenv("MODEL_DIR") + "/20Newsgroups/metadata.json", "w") as output:
        json.dump(metadata, output)


if __name__ == "__main__":
    main()
