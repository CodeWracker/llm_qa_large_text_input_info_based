from datasets import load_dataset
from pprint import pprint
import pandas as pd
from tqdm import tqdm
import json
import pickle

qasper = load_dataset("allenai/qasper")
quality = load_dataset("tasksource/QuALITY")



"""
DatasetDict({
    train: Dataset({
        features: ['id', 'title', 'abstract', 'full_text', 'qas', 'figures_and_tables'],
        num_rows: 888
    })
    validation: Dataset({
        features: ['id', 'title', 'abstract', 'full_text', 'qas', 'figures_and_tables'],
        num_rows: 281
    })
    test: Dataset({
        features: ['id', 'title', 'abstract', 'full_text', 'qas', 'figures_and_tables'],
        num_rows: 416
    })
})
DatasetDict({
    validation: Dataset({
        features: ['article_id', 'set_unique_id', 'batch_num', 'writer_id', 'source', 'title', 'year', 'author', 'topic', 'article', 'url', 'license', 'question', 'question_unique_id', 'options', 'writer_label', 'gold_label', 'validation', 'speed_validation', 'difficult'],
        num_rows: 2086
    })
    train: Dataset({
        features: ['article_id', 'set_unique_id', 'batch_num', 'writer_id', 'source', 'title', 'year', 'author', 'topic', 'article', 'url', 'license', 'question', 'question_unique_id', 'options', 'writer_label', 'gold_label', 'validation', 'speed_validation', 'difficult'],
        num_rows: 2523
    })
})

"""


"""
QASPER DATASET https://huggingface.co/datasets/allenai/qasper
{
  'id': "Paper ID (string)",
  'title': "Paper Title",
  'abstract': "paper abstract ...",
  'full_text': {
      'paragraphs':[["section1_paragraph1_text","section1_paragraph2_text",...],["section2_paragraph1_text","section2_paragraph2_text",...]],
      'section_name':["section1_title","section2_title"],...},
  'qas': {
  'answers':[{
      'annotation_id': ["q1_answer1_annotation_id","q1_answer2_annotation_id"]
      'answer': [{
          'unanswerable':False,
          'extractive_spans':["q1_answer1_extractive_span1","q1_answer1_extractive_span2"],
          'yes_no':False,
          'free_form_answer':"q1_answer1",
          'evidence':["q1_answer1_evidence1","q1_answer1_evidence2",..],
          'highlighted_evidence':["q1_answer1_highlighted_evidence1","q1_answer1_highlighted_evidence2",..]
          },
          {
          'unanswerable':False,
          'extractive_spans':["q1_answer2_extractive_span1","q1_answer2_extractive_span2"],
          'yes_no':False,
          'free_form_answer':"q1_answer2",
          'evidence':["q1_answer2_evidence1","q1_answer2_evidence2",..],
          'highlighted_evidence':["q1_answer2_highlighted_evidence1","q1_answer2_highlighted_evidence2",..]
          }],
      'worker_id':["q1_answer1_worker_id","q1_answer2_worker_id"]
      },{...["question2's answers"]..},{...["question3's answers"]..}],
  'question':["question1","question2","question3"...],
  'question_id':["question1_id","question2_id","question3_id"...],
  'question_writer':["question1_writer_id","question2_writer_id","question3_writer_id"...],
  'nlp_background':["question1_writer_nlp_background","question2_writer_nlp_background",...],
  'topic_background':["question1_writer_topic_background","question2_writer_topic_background",...],
  'paper_read': ["question1_writer_paper_read_status","question2_writer_paper_read_status",...],
  'search_query':["question1_search_query","question2_search_query","question3_search_query"...],
  }
}

"""


"""
QUALITY DATASET https://huggingface.co/datasets/tasksource/QuALITY , https://github.com/nyu-mll/quality/tree/main

- `article_id`: String. A five-digit number uniquely identifying the article. In each split, there are exactly two lines containing the same `article_id`, because two writers wrote questions for the same article.
- `set_unique_id`: String. The unique ID corresponding to the set of questions, which corresponds to the line of json. Each set of questions is written by the same writer.
- `batch_num`: String. The batch number. Our data collection is split in two groups, and there are three batches in each group. `[i][j]` means the j-th batch in the i-th group. For example, `23` corresponds to the third batch in the second group.
- `writer_id`: String. The anonymized ID of the writer who wrote this set of questions. 
- `source`: String. The source of the article. 
- `title`: String. The title of the article.
- `author`: String. The author of the article.
- `topic`: String. The topic of the article.
- `url`: String. The URL of the original unprocessed source article. 
- `year`: String. The (often approximate) publication year of the article. The exact year is often difficult to locate or scrape; in that case, we use (the author's year of birth + the author's year of death) / 2 as the approximate publication year. 
- `license`: String. The license information for the article. 
- `article`: String. The HTML of the article. A script that converts HTML to plain texts is provided. 
- `questions`: A list of dictionaries explained below. Each line of json has a different number of questions because some questions were removed following validation.

As discussed, the value of `questions` is a list of dictionaries. Each dictionary has the following fields.
- `question`: The question. 
- `options`: A list of four answer options.
- `gold_label`: The correct answer, defined by a majority vote of 3 or 5 annotators + the original writer's label. The number corresponds to the option number (1-indexed) in `options`. 
- `writer_label`: The label the writer provided. The number corresponds to the option number (1-indexed) in `options`. 
- `validation`: A list of dictionaries containing the untimed validation results. Each dictionary contains the following fields.
    - `untimed_annotator_id`: The anonymized annotator IDs corresponding to the untimed validation results shown in `untimed_answer`.
    - `untimed_answer`: The responses in the untimed validation. Each question in the training set is annotated by three workers in most cases, and each question in the dev/test sets is annotated by five cases in most cases (see paper for exceptions). 
    - `untimed_eval1_answerability`: The responses (represented numerically) to the first eval question in untimed validation. We asked the raters: "Is the question answerable and unambiguous?" The values correspond to the following choices:
        - 1: Yes, there is a single answer choice that is the most correct.
        - 2: No, two or more answer choices are equally correct.
        - 3: No, it is unclear what the question is asking, or the question or answer choices are unrelated to the passage.
    - `untimed_eval2_context`: The responses (represented numerically) to the second eval question in untimed validation. We asked the raters: "How much of the passage/text is needed as context to answer this question correctly?" The values correspond to the following choices:
        - 1: Only a sentence or two of context.
        - 2: At least a long paragraph or two of context.
        - 3: At least a third of the passage for context.
        - 4: Most or all of the passage for context.
    - `untimed_eval3_distractor`: The responses to the third eval question in untimed validation. We asked the raters: "Which of the options that you did not select was the best "distractor" item (i.e., an answer choice that you might be tempted to select if you hadn't read the text very closely)?" The numbers correspond to the option numbers (1-indexed).
- `speed_validation`: A list of dictionaries containing the speed validation results. Each dictionary contains the following fields.
    - `speed_annotator_id`: The anonymized annotator IDs corresponding to the speed annotation results shown in `speed_answer`.
    - `speed_answer`: The responses in the speed validation. Each question is annotated by five workers.
- `difficult`: A binary value. `1` means that less than 50% of the speed annotations answer the question correctly, so we include this question in the `hard` subset. Otherwise, the value is `0`. In our evaluations, we report one accuracy figure for the entire dataset, and a second for the `difficult=1` subset.

"""

class FreeFormAnswer:
    def __init__(self, free_form_answers, gold_answer = ""):
        self.gold_answer = gold_answer
        if isinstance(free_form_answers, list):
            self.free_form_answers = free_form_answers
        else:
            self.free_form_answers = [str(free_form_answers)]
            
        ans = []
        
        for answer in self.free_form_answers:
            if answer != "":
                ans.append(answer)
        self.free_form_answers = ans
        
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

class Datasetdata:
    def __init__(self, title, abstract, full_text, qas, src_dataset):
        self.title = title
        self.abstract = abstract
        self.full_text = full_text
        self.qas = qas
        self.src_dataset = src_dataset
        
    def to_dict(self):
        qas_dicts = []
        for qa in self.qas:
            qas_dicts.append(qa.to_dict())
        return {
            "title": self.title,
            "abstract": self.abstract,
            "full_text": self.full_text,
            "qas": qas_dicts,
            "src_dataset": self.src_dataset
        }

class JoinDataset:
    def __init__(self):
        self.train_dataset = []
        self.validation_dataset = []
        self.test_dataset = []
    
    def to_dict(self):
        return {
            "train_dataset": [dataset.to_dict() for dataset in self.train_dataset],
            "validation_dataset": [dataset.to_dict() for dataset in self.validation_dataset],
            "test_dataset": [dataset.to_dict() for dataset in self.test_dataset]
        }
    
    def remove_full_text(self):
        for dataset in self.train_dataset:
            dataset.full_text = ""
        for dataset in self.validation_dataset:
            dataset.full_text = ""
        for dataset in self.test_dataset:
            dataset.full_text = ""


joined_dataset = JoinDataset()


def process_qasper(dataset, dataset_name):
    dataset_data = []
    print(f"Processing {dataset_name} dataset - Qasper")
    for i in tqdm(range(len(dataset))):
        title = dataset[i]["title"]
        abstract = dataset[i]["abstract"]
        full_text = ""
        for j in range(len(dataset[i]["full_text"]['paragraphs'])):
            for k in range(len(dataset[i]["full_text"]['paragraphs'][j])):
                full_text += dataset[i]["full_text"]['paragraphs'][j][k]
        qas = []
        questions = dataset[i]["qas"]["question"]
        answers = dataset[i]["qas"]["answers"]
        for j in range(len(questions)):
            question = questions[j]
            unanswerable = answers[j]['answer'][0]["unanswerable"] 
            options = answers[j]['answer']
            free_form_answer = []
            if not unanswerable:
                for k in range(len(options)):
                    yes_no = options[k]['yes_no'] 
                    if yes_no == None:
                        if isinstance(options[k]['free_form_answer'], list):
                            for fr_frm_ans in options[k]['free_form_answer']:
                                free_form_answer.append(fr_frm_ans) 
                        else:
                            free_form_answer.append(options[k]['free_form_answer'])
                        if isinstance(options[k]['extractive_spans'], list):
                            for ext_spans in options[k]['extractive_spans']:
                                free_form_answer.append(ext_spans)
                        else:
                            free_form_answer.append(options[k]['extractive_spans'])
                    else:
                        free_form_answer.append("yes" if yes_no else "no")
            else:
                free_form_answer.append("N/A")
            question_answer = QuestionAnswer(question, unanswerable, free_form_answer)
            qas.append(question_answer)
        dataset_data.append(Datasetdata(title, abstract, full_text, qas, "qasper"))
    return dataset_data

joined_dataset.train_dataset.extend(process_qasper(qasper["train"], "train"))
joined_dataset.validation_dataset.extend(process_qasper(qasper["validation"], "validation"))
joined_dataset.test_dataset.extend(process_qasper(qasper["test"], "test"))

# Quality dataset

def process_quality(dataset, dataset_name):
    dataset_data = []
    print(f"Processing {dataset_name} dataset - Quality")
    for i in tqdm(range(len(dataset))):
        title = dataset[i]["title"]
        full_text = dataset[i]["article"]
        qas = []
        question = dataset[i]["question"]
        options = dataset[i]["options"]
        option_answers = FreeFormAnswer(options)
        qas.append(QuestionAnswer(question, False, option_answers))
        dataset_data.append(Datasetdata(title, "", full_text, qas, "quality"))
    return dataset_data

joined_dataset.train_dataset.extend(process_quality(quality["train"], "train"))
joined_dataset.validation_dataset.extend(process_quality(quality["validation"], "validation"))

# save joined dataset
# \u201d é ", relaxa,o codigo não ta quebrado. Mas se alterar pra salvar como aspas mesmo, ai vai quebrar o json pq vai fechar as aspas no lugar errado
with open("joined_dataset.json", "w") as f:
    json.dump(joined_dataset.to_dict(), f)

# save json without full text
with open("joined_dataset_no_full_text.json", "w") as f:
    joined_dataset_no_full_text = joined_dataset
    joined_dataset_no_full_text.remove_full_text()
    json.dump(joined_dataset_no_full_text.to_dict(), f)

# save as pickle
with open("joined_dataset.pkl", "wb") as f:
    pickle.dump(joined_dataset, f)





