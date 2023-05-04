#https://github.com/santiagxf/mlflow-deployments/blob/main/transformer-classifier/bert-for-classification.ipynb
#https://santiagof.medium.com/effortless-models-deployment-with-mlflow-packing-a-nlp-product-review-classifier-from-huggingface-13be2650333


from transformers.models.auto import AutoConfig, AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer

model_uri = 'nlptown/bert-base-multilingual-uncased-sentiment'
config = AutoConfig.from_pretrained(model_uri)

print('Architecture:', config.architectures)
print('Classes:', config.label2id.keys())

tokenizer = AutoTokenizer.from_pretrained(model_uri)
model = AutoModelForSequenceClassification.from_pretrained(model_uri, config=config)

import torch

if torch.cuda.is_available():
    print("Switching model to CUDA device")
    model = model.cuda()
else:
    print("No CUDA device found. Using CPU.")
    
_ = model.eval()


import pandas as pd 
sample = pd.DataFrame({ 'text': ['good enough',
                                 'The overall quality if good, but there are certain aspects of the product that made it hard to use']})
sample

inputs = tokenizer(list(sample['text'].values), padding=True, return_tensors='pt')

if model.device.index != None:
    print("Model is in a different device as inputs. Moving location to device:", model.device.index)
    for key in inputs.keys():
        inputs[key] = inputs[key].to(model.device.index)
    
predictions = model(**inputs)

import torch
probs = torch.nn.Softmax(dim=1)(predictions.logits)


#We are using PyTorch backend with transformers, which will return tensors in the training/inference device. To easily manipulate them, we can move them to a numpy array:
probs = probs.detach().cpu().numpy()


classes = probs.argmax(axis=1)
confidences = probs.max(axis=1)


outputs = pd.DataFrame({ 'rating': [config.id2label[c] for c in classes], 'confidence': confidences })
outputs


from mlflow.models.signature import infer_signature

signature = infer_signature(sample, outputs)
signature


#Mlflow provides another way to deal with artifacts that you model may need to opperate but that you don't want to serialize in a Python object. That is done by indicating artifacts
model_path = 'rating_classifier'
model.save_pretrained(model_path)

#This will generate a single file called pytorch_model.bin which contains the weights of the model itself. However, remember that in order to run the model we also need it's corresponding tokenizer. The same save_pretrained method is available for the tokenizer, which will generate other set of files:

tokenizer.save_pretrained(model_path)
#Here we can actually see all the files the tokenizer needs in order to operate. Let's tell Mlflow that we need all thes files to run the model. First, we need to create the dictionary I mentioned before:

import os, pathlib

artifacts = { pathlib.Path(file).stem: os.path.join(model_path, file) 
             for file in os.listdir(model_path) 
             if not os.path.basename(file).startswith('.') }


artifacts




from mlflow.pyfunc import PythonModel, PythonModelContext
from typing import Dict

class BertTextClassifier(PythonModel):
    def load_context(self, context: PythonModelContext):
        import os
        from transformers.models.auto import AutoConfig, AutoModelForSequenceClassification
        from transformers.models.auto.tokenization_auto import AutoTokenizer
        
        config_file = os.path.dirname(context.artifacts["config"])
        self.config = AutoConfig.from_pretrained(config_file)
        self.tokenizer = AutoTokenizer.from_pretrained(config_file)
        self.model = AutoModelForSequenceClassification.from_pretrained(config_file, config=self.config)
        
        if torch.cuda.is_available():
            print('[INFO] Model is being sent to CUDA device as GPU is available')
            self.model = self.model.cuda()
        else:
            print('[INFO] Model will use CPU runtime')
        
        _ = self.model.eval()
        
    def _predict_batch(self, data):
        import torch
        import pandas as pd
        
        with torch.no_grad():
            inputs = self.tokenizer(list(data['text'].values), padding=True, return_tensors='pt', truncation=True)
        
            if self.model.device.index != None:
                torch.cuda.empty_cache()
                for key in inputs.keys():
                    inputs[key] = inputs[key].to(self.model.device.index)

            predictions = self.model(**inputs)
            probs = torch.nn.Softmax(dim=1)(predictions.logits)
            probs = probs.detach().cpu().numpy()

            classes = probs.argmax(axis=1)
            confidences = probs.max(axis=1)

            return classes, confidences
        
    def predict(self, context: PythonModelContext, data: pd.DataFrame) -> pd.DataFrame:
        import math
        import numpy as np
        
        batch_size = 64
        sample_size = len(data)
        
        classes = np.zeros(sample_size)
        confidences = np.zeros(sample_size)

        for batch_idx in range(0, math.ceil(sample_size / batch_size)):
            bfrom = batch_idx * batch_size
            bto = bfrom + batch_size
            
            c, p = self._predict_batch(data.iloc[bfrom:bto])
            classes[bfrom:bto] = c
            confidences[bfrom:bto] = p
            
        return pd.DataFrame({'rating': [self.config.id2label[c] for c in classes], 
                             'confidence': confidences })  
        
        
#NOTE: pip3 install mlflow[extras]==1.27.0 
#https://cloudera.slack.com/archives/CCRQ8HRH7/p1682585559662509     

import mlflow
import mlflow.pyfunc
mlflow.set_experiment('bert-classification')

with mlflow.start_run(run_name="marc"):
    mlflow.pyfunc.log_model('classifier', 
                            python_model=BertTextClassifier(), 
                            artifacts=artifacts, 
                            signature=signature,
                            registered_model_name='bert-rating-classification')
      
      



mlflow.set_experiment('bert-classification')
mlflow.start_run

import mlflow
#load_model OT IMPLEMENTED but manual steps available at https://docs.google.com/document/d/1R0I-RBfKAZ4vD_ciSMqEcT6buvU_VK0o5yQhDIMg4pc/edit#
#https://github.com/frischHWC/cml-experiences/blob/main/weather-model-with-ml-flow.py
#model = mlflow.pyfunc.load_model('models:/bert-rating-classification/latest')
#model.predict(sample)



