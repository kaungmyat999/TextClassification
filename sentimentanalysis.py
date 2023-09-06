from  transformers import pipeline
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from util import preprocess_function, tokenizer
from transformers import AutoModelForSequenceClassification
import numpy as np
from datasets import load_metric
from transformers import create_optimizer
import tensorflow as tf

imdb = load_dataset("imdb")

#Selection a few data for fine-tuning (Pytorch Version)
small_train_dataset = imdb["train"].shuffle(seed=42).select([i for i in list(range(3000))])
small_test_dataset = imdb["test"].shuffle(seed=42).select([i for i in list(range(300))])

#ALl Data (TF Version)
tokenized_imdb = imdb.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


#Tokenization

tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)

tf_train_dataset = tokenized_imdb["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "label"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_validation_dataset = tokenized_imdb["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "label"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)


#Selection PreTrained Model 
#Here Distilbert is selected
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)



def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
   load_f1 = load_metric("f1")

   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
   return {"accuracy": accuracy, "f1": f1}


def PytorchVersion():
    from huggingface_hub import notebook_login
    notebook_login()

    from transformers import TrainingArguments, Trainer

    repo_name = "finetuning-sentiment-model-3000-samples"

    training_args = TrainingArguments(
       output_dir=repo_name,
       learning_rate=2e-5,
       per_device_train_batch_size=16,
       per_device_eval_batch_size=16,
       num_train_epochs=2,
       weight_decay=0.01,
       save_strategy="epoch",
       push_to_hub=True,
    )

    trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=tokenized_train,
       eval_dataset=tokenized_test,
       tokenizer=tokenizer,
       data_collator=data_collator,
       compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()
    trainer.push_to_hub()


def TF_Version():
    from transformers import TFAutoModelForSequenceClassification

    batch_size = 16
    num_epochs = 5
    batches_per_epoch = len(tokenized_imdb["train"]) // batch_size
    total_train_steps = int(batches_per_epoch * num_epochs)
    optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

    #Compile
    model.compile(optimizer=optimizer)
    
    #Training
    model.fit(x=tf_train_dataset, validation_data=tf_validation_dataset, epochs=3)
    return model


tfModel = TF_Version()


#Testing With User Prompt

inputStr = ["I don't like that movie","I love it!"]
tokenized = tokenizer(inputStr,truncation=True, padding=True,return_tensors="tf")

predictions = model(tokenized)

#Decoding
predicted_logits = predictions.logits
predicted_probabilities = tf.nn.softmax(predicted_logits, axis=-1)
predicted_labels = tf.argmax(predicted_probabilities, axis=1).numpy()

print("Predicted Labels:", predicted_labels)




