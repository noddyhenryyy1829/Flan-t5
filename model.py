import os
import re
import pathlib
import logging
import shutil
import label_studio_sdk
import dotenv


dotenv.load_dotenv(override=True)
model=None
# dotenv.load_dotenv() 

import torch
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from transformers import pipeline, Pipeline
from itertools import groupby
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer
from transformers import DataCollatorForTokenClassification
from datasets import Dataset, ClassLabel, Value, Sequence, Features
from functools import partial

logger = logging.getLogger(__name__)

class EntityExtractionModel(LabelStudioMLBase):
    """Name Entity Recognition Interactive Model
    """
    LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST', 'http://localhost:8080')
    LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY')
    START_TRAINING_EACH_N_UPDATES = int(os.getenv('START_TRAINING_EACH_N_UPDATES', 10))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 1e-3))
    NUM_TRAIN_EPOCHS = int(os.getenv('NUM_TRAIN_EPOCHS', 100))
    WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY', 0.01))

    MODEL_DIR = os.getenv('MODEL_DIR', './results/entity-extraction')
    BASELINE_MODEL_NAME = os.getenv('BASELINE_MODEL_NAME', 'dslim/bert-base-NER')
    FINETUNED_MODEL_NAME = os.getenv('FINETUNED_MODEL_NAME', 'finetuned_model')
    CURRENT_MODEL_VERSION=os.getenv('CURRENT_MODEL_VERSION', '1.0.0')
    DEVICE = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_LOAD_MODE=None


    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", self.CURRENT_MODEL_VERSION)

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        logger.info(f"Task:{tasks}")
        li = self.label_interface
        from_name, to_name, value = li.get_first_tag_occurence('Labels', 'Text')
        texts = [self.preload_task_data(task, task['data'][value]) for task in tasks]

        return self.prediction(texts,context,**kwargs)
    

    def predict_external(self, texts, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param texts: [List of text in JSON format]
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        return self.prediction(texts,None,**kwargs)
        
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """
        global model

        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
            logger.info(f"Skip training: event {event} is not supported")
            return

        project_id = data['project']['id']
        tasks = self._get_tasks(project_id)

        if len(tasks) % self.START_TRAINING_EACH_N_UPDATES != 0 and event != 'START_TRAINING':
            logger.info(f"Skip training: {len(tasks)} tasks are not multiple of {self.START_TRAINING_EACH_N_UPDATES}")
            return

        # we need to convert Label Studio NER annotations to hugingface NER format in datasets
        # for example:
        # {'id': '0',
        #  'ner_tags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
        #  'tokens': ['@paulwalk', 'It', "'s", 'the', 'view', 'from', 'where', 'I', "'m", 'living', 'for', 'two', 'weeks', '.', 'Empire', 'State', 'Building', '=', 'ESB', '.', 'Pretty', 'bad', 'storm', 'here', 'last', 'evening', '.']
        # }
        ds_raw = []
        from_name, to_name, value = self.label_interface.get_first_tag_occurence('Labels', 'Text')
        tokenizer = AutoTokenizer.from_pretrained(self.getModelPath())

        no_label = 'O'
        label_to_id = {no_label: 0}
        for task in tasks:
            for annotation in task['annotations']:
                if not annotation.get('result'):
                    continue
                spans = [{'label': r['value']['labels'][0], 'start': r['value']['start'], 'end': r['value']['end']} for r in annotation['result']]
                spans = sorted(spans, key=lambda x: x['start'])
                text = self.preload_task_data(task, task['data'][value])

                # insert tokenizer.pad_token to the unlabeled chunks of the text in-between the labeled spans, as well as to the beginning and end of the text
                last_end = 0
                all_spans = []
                for span in spans:
                    if last_end < span['start']:
                        all_spans.append({'label': no_label, 'start': last_end, 'end': span['start']})
                    all_spans.append(span)
                    last_end = span['end']
                if last_end < len(text):
                    all_spans.append({'label': no_label, 'start': last_end, 'end': len(text)})

                # now tokenize chunks separately and add them to the dataset
                item = {'id': task['id'], 'tokens': [], 'ner_tags': []}
                for span in all_spans:
                    tokens = tokenizer.tokenize(text[span['start']:span['end']])
                    item['tokens'].extend(tokens)
                    if span['label'] == no_label:
                        item['ner_tags'].extend([label_to_id[no_label]] * len(tokens))
                    else:
                        label = 'B-' + span['label']
                        if label not in label_to_id:
                            label_to_id[label] = len(label_to_id)
                        item['ner_tags'].append(label_to_id[label])
                        if len(tokens) > 1:
                            label = 'I-' + span['label']
                            if label not in label_to_id:
                                label_to_id[label] = len(label_to_id)
                            item['ner_tags'].extend([label_to_id[label] for _ in range(1, len(tokens))])
                ds_raw.append(item)

        logger.debug(f"Dataset: {ds_raw}")
        # convert to huggingface dataset
        # Define the features of your dataset
        features = Features({
            'id': Value('string'),
            'tokens': Sequence(Value('string')),
            'ner_tags': Sequence(ClassLabel(names=list(label_to_id.keys())))
        })
        hf_dataset = Dataset.from_list(ds_raw, features=features)
        tokenized_dataset = hf_dataset.map(partial(self.tokenize_and_align_labels, tokenizer=tokenizer), batched=True)

        logger.debug(f"HF Dataset: {tokenized_dataset}")

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        id_to_label = {i: label for label, i in label_to_id.items()}
        logger.debug(f"Labels: {id_to_label}")
        logger.info(id_to_label)

        _model = AutoModelForTokenClassification.from_pretrained(
            self.getModelPath(), num_labels=len(id_to_label),
            id2label=id_to_label, label2id=label_to_id,
            ignore_mismatched_sizes=True)
        logger.debug(f"Model: {_model}")

        chk_path = self.getModelPath('Finetunned')

        training_args = TrainingArguments(
            output_dir=chk_path,
            learning_rate=self.LEARNING_RATE,
            per_device_train_batch_size=8,
            num_train_epochs=self.NUM_TRAIN_EPOCHS,
            weight_decay=self.WEIGHT_DECAY,
            evaluation_strategy="no",
            save_strategy='no',
            logging_dir=None,
            logging_strategy='no',
            report_to=["none"]
        )

        trainer = Trainer(
            model=_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        trainer.train()

      

        logger.info(f"Model is trained and saved as {chk_path}")
        trainer.save_model(chk_path)
        tokenizer.save_pretrained(chk_path)
        logger.info(_model.config.id2label)
        # self.updateEnvFile("CURRENT_MODEL_VERSION",new_Model_version)
        # self.CURRENT_MODEL_VERSION = os.getenv("CURRENT_MODEL_VERSION", self.CURRENT_MODEL_VERSION)
        model = None


    def getModelPath(self,mode='Base',finetunedVersion=None):
        if mode == 'Base':
                return self.BASELINE_MODEL_NAME # str(pathlib.Path(self.MODEL_DIR)/self.BASELINE_MODEL_NAME)
        elif mode == 'Finetunned':
            # return str(pathlib.Path(self.MODEL_DIR)/self.FINETUNED_MODEL_NAME/(finetunedVersion if finetunedVersion is not None else self.CURRENT_MODEL_VERSION))
            return str(pathlib.Path(self.MODEL_DIR)/self.FINETUNED_MODEL_NAME)
        else:
            return None
    
    def loadModel(self,mode='Base'):
        chk_path = self.getModelPath(mode,self.CURRENT_MODEL_VERSION)
        logger.info(f"Loading {mode.lower()} model from {chk_path}")
        return pipeline("ner", model=chk_path, tokenizer=chk_path)

    def lazyInit(self):
        global model
        if model is None: 
            try:
                model = self.loadModel('Finetunned')
                self.MODEL_LOAD_MODE = 'Finetunned'
            except Exception as e:
                logger.error(str(e), exc_info=True)
                model = self.loadModel()
                self.MODEL_LOAD_MODE = 'Base'

    def getNextVersion(self, model_dir, bump='minor'):
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            return '1.0.0'

        versions = [d for d in os.listdir(model_dir) if re.match(r'\d+\.\d+\.\d+', d)]
        versions.sort(key=lambda v: list(map(int, v.split('.'))))
        
        last_version = versions[-1]
        major, minor, patch = map(int, last_version.split('.'))

        if bump == 'patch':
            patch += 1
        elif bump == 'minor':
            minor += 1
            patch = 0
        elif bump == 'major':
            major += 1
            minor = patch = 0

        return f'{major}.{minor}.{patch}'

    def updateEnvFile(self, key, value ):
        # Load existing values
        env_path= dotenv.find_dotenv()
        dotenv.load_dotenv(env_path)

        dotenv.set_key(env_path,key, value)

        dotenv.load_dotenv(env_path, override=True)

        self.CURRENT_MODEL_VERSION = os.getenv("CURRENT_MODEL_VERSION", self.CURRENT_MODEL_VERSION)

    def _get_tasks(self, project_id):
        # download annotated tasks from Label Studio
        ls = label_studio_sdk.Client(self.LABEL_STUDIO_HOST, self.LABEL_STUDIO_API_KEY)
        project = ls.get_project(id=project_id)
        tasks = project.get_labeled_tasks()
        return tasks

    def tokenize_and_align_labels(self, examples, tokenizer):
        """
        From example https://huggingface.co/docs/transformers/en/tasks/token_classification#preprocess
        """
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs


    def prediction(self, texts:str, context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param text: text to be available for prediction
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        self.lazyInit()
        logger.info(model.model.config.id2label)
        li = self.label_interface
        from_name, to_name, value = li.get_first_tag_occurence('Labels', 'Text')

        # run predictions
        model_predictions = model(texts)
        logger.info(model_predictions)

        predictions = []
        for prediction in model_predictions:
            # prediction returned in the format: [{'entity': 'B-ORG', 'score': 0.999, 'index': 1, 'start': 0, 'end': 7, 'word': 'Google'}, ...]
            # we need to group them by 'B-' and 'I-' prefixes to form entities
            results = []
            avg_score = 0
            for label, group in groupby(prediction, key=lambda x: re.sub(r'^[BI]-', '', x['entity'])):
                entities = list(group)
                # start = entities[0]['start']
                # end = entities[-1]['end']
                # score = float(sum([entity['score'] for entity in entities]) / len(entities))
                entities_list =[]
                entity_list_item=None
                for entity in entities:
                    if entity['entity'].startswith('B-'):
                        if not entity['word'].startswith("##"):
                            if entity_list_item is not None and len(entity_list_item)>0:
                                entities_list.append(entity_list_item)
                            entity_list_item=[]
                    
                    entity_list_item.append(entity)
                if entity_list_item is not None and len(entity_list_item)>0:
                    entities_list.append(entity_list_item)
                
                for entity_items in entities_list:
                    start = entity_items[0]['start']
                    end = entity_items[-1]['end']
                    score = float(sum([entity['score'] for entity in entity_items]) / len(entity_items))

                    results.append({
                        'from_name': from_name,
                        'to_name': to_name,
                        'type': 'labels',
                        'value': {
                            'start': start,
                            'end': end,
                            'labels': [label],
                            'text':texts[0][start:end]
                        },
                        'score': score
                    })
                    avg_score += score
            if results:
                predictions.append({
                    'result': results,
                    'score': avg_score / len(results),
                    'model_version': self.get('model_version')
                })
        
        return ModelResponse(predictions=predictions, model_version=self.get('model_version'))

def prediction_old(self, texts:str, context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param text: text to be available for prediction
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        self.lazyInit()
        logger.info(model.model.config.id2label)
        li = self.label_interface
        from_name, to_name, value = li.get_first_tag_occurence('Labels', 'Text')

        # run predictions
        model_predictions = model(texts)
        logger.info(model_predictions)

        predictions = []
        for prediction in model_predictions:
            # prediction returned in the format: [{'entity': 'B-ORG', 'score': 0.999, 'index': 1, 'start': 0, 'end': 7, 'word': 'Google'}, ...]
            # we need to group them by 'B-' and 'I-' prefixes to form entities
            results = []
            avg_score = 0
            for label, group in groupby(prediction, key=lambda x: re.sub(r'^[BI]-', '', x['entity'])):
                entities = list(group)
                start = entities[0]['start']
                end = entities[-1]['end']
                score = float(sum([entity['score'] for entity in entities]) / len(entities))
                results.append({
                    'from_name': from_name,
                    'to_name': to_name,
                    'type': 'labels',
                    'value': {
                        'start': start,
                        'end': end,
                        'labels': [label]
                    },
                    'score': score
                })
                avg_score += score
            if results:
                predictions.append({
                    'result': results,
                    'score': avg_score / len(results),
                    'model_version': self.get('model_version')
                })
        
        return ModelResponse(predictions=predictions, model_version=self.get('model_version'))