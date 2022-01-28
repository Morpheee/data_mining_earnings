#! /usr/bin/python3

# source : https://iq.opengenus.org/binary-text-classification-bert/

########################################
### imports ###
########################################

print("\n\t**** start : imports ****\n")

import seaborn as sns
from matplotlib import pyplot as plt
import random
import time
import datetime
import pandas as pd
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, \
    get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import torch
import tensorflow as tf
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# initialize tqdm
tqdm.pandas()

def main(path_train="../data/dataframe_train.csv",
         path_test="../data/dataframe.csv",
         epochs=2,
         batch_size_train=2,
         batch_size_test=2):

    ########################################
    ### using GPU ###
    ########################################
    
    print("\n\t**** start : using GPU ****\n")
    ## tensoflow ##
    device_name = tf.test.gpu_device_name()
    if device_name == '/device:GPU:0':
        print('Tensorflow : Found GPU at: {}'.format(device_name))
    else:
        raise SystemError('Tensorflow : GPU device not found')
    ## pytorch ##
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('PyTorch : There are %d GPU(s) available.' % torch.cuda.device_count())
        print('PyTorch : We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('PyTorch : No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    ########################################
    ### import datasets ###
    ########################################
    
    print("\n\t**** start : import datasets ****\n")

    df = pd.read_csv(path_train)
    sentences = df["text"].values
    labels = df["label"].values
    del df

    ########################################
    ### sentences to IDs and padding ###
    ########################################
    
    print("\n\t**** start : sentences to IDs and padding ****\n")
    ## tokenizer ##

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    ## convert sentences ##

    input_ids = []
    i, n = 0, len(sentences)
    for sent in sentences:
        i = i + 100 / n
        print("{:.2f} %".format(i), end="\r")
        encoded_sent = tokenizer.encode(
            sent,
            add_special_tokens=True,
        )
        input_ids.append(encoded_sent)

    ## padding & Truncating to fit max length ##

    MAX_LEN = 512
    input_ids = list(pad_sequences(
        input_ids,
        maxlen=MAX_LEN,
        dtype="long",
        value=0,
        truncating="post",
        padding="post"
    ))

    ########################################
    ### attention masks ###
    ########################################
    
    print("\n\t**** start : attention masks ****\n")

    attention_masks = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    # 1 when actual token, 0 when padding

    ########################################
    ### train test split ###
    ########################################
    
    print("\n\t**** start : train test split ****\n")

    train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks = train_test_split(
        input_ids, labels, attention_masks,
        random_state=42, test_size=0.1)

    # changing the numpy arrays into tensors for working on GPU.
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)

    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)

    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size_train)

    # DataLoader for our validation(test) set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size_train)

    ########################################
    ### train classifier model ###
    ########################################
    
    print("\n\t**** start : train classifier model ****\n")

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    # run on GPU with pytorch
    model.cuda()

    ## print out the model ##
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())
    print('The BERT model has {:} different named parameters.\n'.format(len(params)))
    print('==== Embedding Layer ====\n')
    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n==== First Transformer ====\n')
    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n==== Output Layer ====\n')
    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    ########################################
    ### optimizer ###
    ########################################
    
    print("\n\t**** start : optimizer ****\n")
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,
                      eps=1e-8
                      )
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    ########################################
    ### training ###
    ########################################

    print("\n\t**** start : training ****\n")

    def flat_accuracy(preds, labels):
        '''
        Calculate the accuracy of our predictions vs labels
        '''
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)


    def flat_auc(preds, labels):
        '''
        Calculate the Area Under the ROC Curve
        More suited to unbalanced data
        '''
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        #switch last label and prediction to avoid "only one class" error.
        if sum(labels_flat) == len(labels_flat) :
            print(labels_flat)
            print(pred_flat)
            labels_flat[-1] = (labels_flat[-1]+1)%2
            print(labels_flat)
            pred_flat[-1] = (pred_flat[-1]+1)%2
            print(pred_flat)
        return roc_auc_score(labels_flat, pred_flat)


    def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))


    # source : huggingface `run_glue.py` at :
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    loss_values = []
    accuracy_values = []
    auc_values = []
    time_values = []

    for epoch_i in range(0, epochs):
        training_time = time.time()
        # ========================================
        #               Training
        # ========================================
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        total_loss = 0
        # set model to train mod
        model.train()
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            # As we unpack the batch, we'll also copy each tensor to the GPU
            # batch  [0]: input ids ; [1]: attention masks ; [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            # clear any previously calculated gradients before performing backward pass.
            model.zero_grad()
            # Perform a forward pass
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            loss = outputs[0]
            total_loss += loss.item()
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # Clip the norm of the gradients to 1.0. Help prevent "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient.
            optimizer.step()
            # Update the learning rate.
            scheduler.step()
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        # ========================================
        #               Validation
        # ========================================
        print("")
        print("Running Validation...")
        t0 = time.time()
        # Put the model in evaluation mode (change dropout behaviour)
        model.eval()
        # Tracking variables
        eval_loss, eval_accuracy, eval_auc = 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and time
            with torch.no_grad():
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)
            # The "logits" are the output values prior to applying an activation function like the softmax.
            logits = outputs[0]
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            # Calculate the accuracy and the AUC for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            tmp_eval_auc = flat_auc(logits, label_ids)
            # Accumulate the total accuracy and auc.
            eval_accuracy += tmp_eval_accuracy
            eval_auc += tmp_eval_auc
            nb_eval_steps += 1
        accuracy_values.append(eval_accuracy / nb_eval_steps)
        auc_values.append(eval_auc / nb_eval_steps)
        time_values.append(time.time() - training_time)
        print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
        print("  AUC: {0:.2f}".format(eval_auc / nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
    print("")
    print("Training complete!")

    ########################################
    ### plots ###
    ########################################
    
    print("\n\t**** start : plots ****\n")

    plt.figure(figsize=(10, 20))
    plt.subplot(2, 2, 1)
    sns.lineplot(time_values, "b-o")
    plt.xticks([0, len(time_values)])
    # plt.xlabel("epoch")
    plt.ylabel("time (s)")
    plt.title("Elapsed time by epoch.")

    plt.subplot(2, 2, 2)
    sns.lineplot(loss_values, "b-o")
    plt.xticks([0, len(loss_values)])
    # plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("CrossEntropy loss")

    plt.subplot(2, 2, 3)
    sns.lineplot(accuracy_values, "b-o")
    plt.xticks([0, len(accuracy_values)])
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy")

    plt.subplot(2, 2, 4)
    sns.lineplot(auc_values, "b-o")
    plt.xticks([0, len(auc_values)])
    plt.xlabel("epoch")
    plt.ylabel("AUC")
    plt.title("Area Under the ROC Curve")

    plt.suptitle("Training BERT")
    plt.tight_layout()
    plt.savefig("../images/bert_loss_acc_auc_time_train.jpg")

    ########################################
    ### test set ###
    ########################################
    
    print("\n\t**** start : test set ****\n")
    ## data preparation ##
    # load dataset
    df = pd.read_csv(path_test)
    print('Number of test sentences: {:,}\n'.format(df.shape[0]))
    sentences = df["text"].values
    labels = df["label"].values
    input_ids = []
    # IDs
    for sent in sentences:
        encoded_sent = tokenizer.encode(
            sent,
            add_special_tokens=True,
        )
        input_ids.append(encoded_sent)
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN,
                              dtype="long", truncating="post", padding="post")
    # attention mask
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # Convert to tensors.
    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)
    # Create DataLoader.
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size_test)

    ## evaluate ##
    print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))
    # model to evaluation mode
    model.eval()
    # Tracking variables
    predictions, true_labels = [], []
    # Predict
    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.append(logits)
        true_labels.append(label_ids)
    print('    DONE.')
    # matthews metric
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
    print('MCC: %.3f' % mcc)
    # accuracy metric
    accuracy_test = np.sum(flat_predictions == flat_true_labels) / len(flat_true_labels)
    print('accuracy : %.3f' % accuracy_test)
    # AUC metric
    auc_test = roc_auc_score(flat_predictions, flat_true_labels)
    print('auc : %3.f' % auc_test)
    # save metrics test
    df_metrics_test = pd.DataFrame({"accuracy": accuracy_test,
                                    "AUC": auc_test,
                                    "MCC": mcc})
    df.to_csv("./metrics_on_test_set.csv")

    ########################################
    ### save the model ###
    ########################################
    
    print("\n\t**** start : save the model ****\n")
    ## save ##
    import os
    output_dir = './model_save/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Saving model to %s" % output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    ## load ##
    # # Load a trained model and vocabulary that you have fine-tuned
    # model = model_class.from_pretrained(output_dir)
    # tokenizer = tokenizer_class.from_pretrained(output_dir)
    # # Copy the model to the GPU.
    # model.to(device)

if __name__=="__main__":
    main()