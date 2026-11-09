# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import re
from nltk.stem import PorterStemmer


class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str) -> tuple[str, float]:
        with torch.no_grad():
            # tokenize the premise and hypothesis
            inputs = self.tokenizer(
                premise, hypothesis, return_tensors='pt', truncation=True,
                max_length=self.model.config.max_position_embeddings, padding=True,
            )
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            # get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        label_id = logits.argmax(dim=-1).item()
        label = self.model.config.id2label[label_id]
        entailment_id = self.model.config.label2id["entailment"]
        entailment_probability = logits.softmax(dim=-1)[0, entailment_id].item()

        # release the tensors and return only the label and a python number.
        del inputs, outputs, logits

        return label, entailment_probability


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    def __init__(self, threshold=0.75, use_stemming=True):
        # At least 75% of the fact words must match.
        self.threshold = threshold
        self.use_stemming = use_stemming
        self.stemmer = PorterStemmer()

        # Skip common words. Keep words like "not" and "no".
        self.stop_words = set(
            "a an the and or but is are was were be been being am to of in on at "
            "for from by with as that this these those it its he she his her "
            "they their them has have had do does did s".split()
        )

    def get_words(self, text: str) -> set:
        text = text.lower()

        # Remove tags like <s>.
        text = re.sub(r"<[^>]+>", " ", text)

        # Find the words and numbers in the text.
        words = re.findall(r"[a-z0-9]+", text)
        useful_words = set()

        # Keep useful words. A set counts each word only once.
        for word in words:
            if word not in self.stop_words:
                if self.use_stemming:
                    word = self.stemmer.stem(word)
                useful_words.add(word)

        return useful_words

    def get_overlap_score(self, fact: str, passages: List[dict]) -> float:
        """Return the fraction of fact words found in the passages."""
        fact_words = self.get_words(fact)

        # No words left? Avoid dividing by zero.
        if not fact_words:
            return 0.0

        passage_words = set()

        # Collect the words from all evidence passages.
        for passage in passages:
            words = self.get_words(passage["text"])
            passage_words.update(words)

        # Count how many fact words appear in the evidence.
        matching_word_count = 0

        for word in fact_words:
            if word in passage_words:
                matching_word_count += 1

        fact_word_count = len(fact_words)

        # Example: 3 matching words out of 4 gives 0.75.
        score = matching_word_count / fact_word_count

        return score

    def predict(self, fact: str, passages: List[dict]) -> str:
        score = self.get_overlap_score(fact, passages)

        # Enough matching words means supported.
        if score >= self.threshold:
            return "S"

        return "NS"


class EntailmentFactChecker(FactChecker):
    minimum_overlap = 0.15

    def __init__(self, ent_model):
        self.ent_model = ent_model
        self.overlap_checker = WordRecallThresholdFactChecker(use_stemming=True)
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")

    def evidence_sentences(self, passages: List[dict], paired=False):
        for passage in passages:
            previous = None
            # keep the boundaries between marked sections of the passage.
            for section in passage["text"].replace("<s>", "").split("</s>"):
                for sentence in self.nlp(section).sents:
                    premise = sentence.text.strip()
                    if not premise:
                        continue
                    if paired and previous is not None:
                        yield previous + " " + premise
                    if not paired:
                        yield premise
                    previous = premise

    def score_evidence(self, fact: str, candidates):
        for premise in candidates:
            overlap = self.overlap_checker.get_overlap_score(fact, [{"text": premise}])
            if overlap >= self.minimum_overlap:
                label, probability = self.ent_model.check_entailment(premise, fact)
                yield premise, label, probability

    def predict(self, fact: str, passages: List[dict]) -> str:
        # try individual sentences first, then try pairs for more context.
        for paired in (False, True):
            candidates = self.evidence_sentences(passages, paired=paired)
            for _, label, _ in self.score_evidence(fact, candidates):
                if label == "entailment":
                    return "S"
        return "NS"


# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations
