def normalize(doc, nlp, lemma=True):
    """

    :param doc: (str)
    :return: list of relevant tokens
    """
    # lowercase document
    doc = doc.lower()
    # apply spacy tokenizer
    doc = nlp(doc)
    # lemmatization, keep alpha numeric tokens, remove stop words
    if lemma:
        tokens = [token.lemma_ for token in doc
                  if token.is_alpha
                  if not token.is_stop]
    else:
        tokens = [token.text for token in doc
                  if token.is_alpha
                  if not token.is_stop]
    return tokens
