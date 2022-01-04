from pathlib import Path
from typing import Union

import mlflow.pyfunc

from annotate import annotate_raw_texts
from Corpus import Corpus


class TopicModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, topic_model: "TopicModel"):
        self.topic_model = topic_model
        assert hasattr(self.topic_model, "fit")

    def fit(self, train_corpus: "Corpus", nlp: "Language"):
        self.topic_model.fit(train_corpus=train_corpus, nlp=nlp)

    def predict(self, context: dict, model_input: Union[str, Path]):
        """

        :param context: {"nlp": "Language", "custom_vocab": dict, "config": dict}
        :param model_input: path to directory containing raw texts to make predictions on
        :return:
        """
        config = context["config"]
        nlp = context["nlp"]

        docbin = annotate_raw_texts(root_dir=model_input, nlp=nlp, config=config)
        # save docbin
        docbin_path = model_input / "preds" / "predict.spacy"
        docbin.to_disk(docbin_path)

        # corpus
        corpus = Corpus(path=docbin_path, custom_vocab=context["custom_vocab"],
                        tfidf_threshold=float(config['model_params']['tfidf_threshold']))
        z_hat, q_theta_hat = self.topic_model.predict_corpus(corpus=corpus, nlp=nlp)
        return z_hat, q_theta_hat