# opens results/joined_dataset.plk and checks if there is a duplicated title. prints out if yes. print all duplicated

"""
{
  "dataset": [
    {
      "title": "Minimally Supervised Learning of Affective Events Using Discourse Relations",
      "abstract": "Recognizing affective events that trigger positive or negative sentiment has a wide range of natural language processing applications but remains a challenging problem mainly because the polarity of an event is not necessarily predictable from its constituent words. In this paper, we propose to propagate affective polarity using discourse relations. Our method is simple and only requires a very small seed lexicon and a large raw corpus. Our experiments using Japanese data show that our method learns affective events effectively without manually labeled data. It also improves supervised learning results when labeled data are small.",
      "full_text": "",
      "qas": [
        {
          "question": "What is the seed lexicon?",
          "unanswerable": false,
          "option_answers": {
            "gold_answer": "",
            "free_form_answers": [
              "a vocabulary of positive and negative predicates that helps determine the polarity score of an event",
              "seed lexicon consists of positive and negative predicates"
            ]
          }
        },
        {
          "question": "What are the results?",
          "unanswerable": false,
          "option_answers": {
            "gold_answer": "",
            "free_form_answers": [
              "Using all data to train: AL -- BiGRU achieved 0.843 accuracy, AL -- BERT achieved 0.863 accuracy, AL+CA+CO -- BiGRU achieved 0.866 accuracy, AL+CA+CO -- BERT achieved 0.835, accuracy, ACP -- BiGRU achieved 0.919 accuracy, ACP -- BERT achived 0.933, accuracy, ACP+AL+CA+CO -- BiGRU achieved 0.917 accuracy, ACP+AL+CA+CO -- BERT achieved 0.913 accuracy. \nUsing a subset to train: BERT achieved 0.876 accuracy using ACP (6K), BERT achieved 0.886 accuracy using ACP (6K) + AL, BiGRU achieved 0.830 accuracy using ACP (6K), BiGRU achieved 0.879 accuracy using ACP (6K) + AL + CA + CO."
            ]
          }
        },
        {
          "question": "How are relations used to propagate polarity?",
          "unanswerable": false,
          "option_answers": {
            "gold_answer": "",
            "free_form_answers": [
              "based on the relation between events, the suggested polarity of one event can determine the possible polarity of the other event ",
              "cause relation: both events in the relation should have the same polarity; concession relation: events should have opposite polarity"
            ]
          }
        },
    ]
}
"""

import pickle



class FreeFormAnswer:
    def __init__(self, free_form_answers, gold_answer=""):
        self.gold_answer = gold_answer
        # Garante que free_form_answers seja uma lista de strings
        if isinstance(free_form_answers, list):
            self.free_form_answers = [str(ans) for ans in free_form_answers if ans != ""]
        else:
            self.free_form_answers = [str(free_form_answers)] if free_form_answers != "" else []

    def to_dict(self):
        return {
            "gold_answer": self.gold_answer,
            "free_form_answers": self.free_form_answers
        }


class QuestionAnswer:
    def __init__(self, question, unanswerable, option_answers):
        self.question = question
        self.unanswerable = unanswerable
        if isinstance(option_answers, FreeFormAnswer):
            self.option_answers = option_answers
        else:
            self.option_answers = FreeFormAnswer(option_answers)

    def to_dict(self):
        return {
            "question": self.question,
            "unanswerable": self.unanswerable,
            "option_answers": self.option_answers.to_dict()
        }


class DatasetData:
    def __init__(self, title, abstract, full_text, qas, src_dataset):
        self.title = title
        self.abstract = abstract
        self.full_text = full_text
        self.qas = qas
        self.src_dataset = src_dataset

    def to_dict(self):
        return {
            "title": self.title,
            "abstract": self.abstract,
            "full_text": self.full_text,
            "qas": [qa.to_dict() for qa in self.qas],
            "src_dataset": self.src_dataset
        }


class JoinedDataset:
    def __init__(self):
        self.dataset = []

    def to_dict(self):
        return {
            "dataset": [data.to_dict() for data in self.dataset]
        }

    def remove_full_text(self):
        for data in self.dataset:
            data.full_text = ""



joined_dataset = pickle.load(open("results/joined_dataset.pkl", "rb"))
unique_titles = []

for data in joined_dataset.dataset:
    title = data.title
    if title in unique_titles:
        print(f"Duplicated title found: {title}")
    else:
        unique_titles.append(title)
        
    
