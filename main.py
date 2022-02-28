
import numpy as np
import os
import pandas as pd
import pathlib
import random
import sys
import torch
import torch.nn as nn
from types import SimpleNamespace

from data_loading import Vocabulary, batch_generator, load_preprocessed_dataset
from docopt_parser import parse_args
from local_logger import LocalLogger
from model import CBOW
from wandb_logger import WandbLogger


ACCUMULATE_GRADS = True
PROCESS_BATCH_SIZE = 7700

PROJECT_NAME = "SLPDL_WordVectors"
ENTITY = "slpdl2022"

DATASET_VERSION = 'ca-100'
#COMPETITION_ROOT = '../input/vectors3'
COMPETITION_ROOT = 'data/word_vectors' # I downloaded it from https://www.kaggle.com/c/vectors3/data
#DATASET_ROOT = f'../input/text-preprocessing/data/{DATASET_VERSION}'
DATASET_ROOT = f'data/text-preprocessing/data/{DATASET_VERSION}'
#WORKING_ROOT = f'data/{DATASET_VERSION}'
MODELS_ROOT = f'model_parameters/{DATASET_VERSION}'
OUTPUTS_ROOT = f'outputs/{DATASET_VERSION}'
DATASET_PREFIX = 'ca.wiki'


params = SimpleNamespace(
    embedding_dim = 100,
    window_size = 7,
    batch_size = 1000,
    epochs = 4,
    preprocessed = f'{DATASET_ROOT}/{DATASET_PREFIX}',
    #working = f'{WORKING_ROOT}/{DATASET_PREFIX}',
    working = f'{OUTPUTS_ROOT}/{DATASET_PREFIX}', # Not used anywhere
    #modelname = f'{WORKING_ROOT}/{DATASET_VERSION}.pt',
    modelname = f'{MODELS_ROOT}/{DATASET_VERSION}.pt',
    train = True # Not used anywhere
)


def train(model, criterion, optimizer, idata, target, batch_size, device, local_logger, log=False):
    model.train()
    local_logger.new_epoch()

    process_batch_size = PROCESS_BATCH_SIZE if ACCUMULATE_GRADS else batch_size
    model.zero_grad()
    proceced_samples = 0

    for X, y in batch_generator(idata, target, process_batch_size, shuffle=True):
        # Get input and target sequences from batch
        X = torch.tensor(X, dtype=torch.long, device=device)
        y = torch.tensor(y, dtype=torch.long, device=device)

        if not ACCUMULATE_GRADS : model.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        if not ACCUMULATE_GRADS : optimizer.step()

        # Training statistics
        local_logger.update_epoch_log(output, y, loss, VERBOSE=True)

        # If accumulation of gradient to have bigger batches, ugly solution but should work
        proceced_samples += process_batch_size
        if ACCUMULATE_GRADS and proceced_samples >= batch_size:
            proceced_samples = 0
            optimizer.step()
            model.zero_grad()


    local_logger.finish_epoch(VERBOSE=log)
    return local_logger['accuracy'][-1], local_logger['total_loss'][-1]

def validate(model, criterion, idata, target, batch_size, device, local_logger=None):
    model.eval()
    if local_logger is not None : local_logger.new_epoch()

    if ACCUMULATE_GRADS : batch_size = PROCESS_BATCH_SIZE

    y_pred = []
    with torch.no_grad():
        for X, y in batch_generator(idata, target, batch_size, shuffle=False):
            # Get input and target sequences from batch
            X = torch.tensor(X, dtype=torch.long, device=device)
            output = model(X)

            if target is not None:
                y = torch.tensor(y, dtype=torch.long, device=device)
                loss = criterion(output, y)
                
                local_logger.update_epoch_log(output, y, loss, VERBOSE=False)
            else:
                pred = torch.max(output, 1)[1].detach().to('cpu').numpy()
                y_pred.append(pred)

    if target is not None:
        local_logger.finish_epoch(VERBOSE=False)
        return local_logger['accuracy'][-1], local_logger['total_loss'][-1]

    else:
        return np.concatenate(y_pred)

def build_periodico_dataset(vocab):
    # IGNSAI: Funcion un poco fea porque depende directamente de constantes o hard-wired text

    # 'El Periodico' validation dataset
    valid_x_df = pd.read_csv(f'{COMPETITION_ROOT}/x_valid.csv')
    tokens = valid_x_df.columns[1:]
    valid_x = valid_x_df[tokens].apply(vocab.get_index).to_numpy(dtype='int32')
    valid_y_df = pd.read_csv(f'{COMPETITION_ROOT}/y_valid.csv')
    valid_y = valid_y_df['token'].apply(vocab.get_index).to_numpy(dtype='int32')

    # 'El Periodico' test dataset
    valid_x_df = pd.read_csv(f'{COMPETITION_ROOT}/x_test.csv')
    test_x = valid_x_df[tokens].apply(vocab.get_index).to_numpy(dtype='int32')

    return valid_x, valid_y, valid_x_df, test_x

def print_model(model):
    print(model)
    for name, param in model.named_parameters():
        print(f'{name:20} {param.numel()} {list(param.shape)}')
    print(f'TOTAL{" " * 16}{sum(p.numel() for p in model.parameters())}')

def build_optimizer(optimizer_class, model, **optimizer_params):
    # TODO: hyperparam optimization (optimizer & learning_rate including time/space dependant learning_rates)
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    return optimizer

def main(window_size, embedding_dim,  weights, vector, train_weights, shared_embedding, num_epochs, batch_size, fract, fract_dataset, lr, preprocessed_path, modelname, experiment_name, device):

    ####################################### PRETRAINING  #######################################
    vocab, data = load_preprocessed_dataset(preprocessed_path)
    valid_x, valid_y, valid_x_df, test_x = build_periodico_dataset(vocab)

    if fract:
        batch_size = int(len(data[0][0]) // fract_dataset)

    # TODO: Section E, modify model to improve the embeddings.
    model = CBOW(len(vocab), embedding_dim, num_context_words=window_size-1, weights=weights, vector=vector, train_weights=train_weights, shared_embedding=shared_embedding).to(device)
    print_model(model)

    criterion = nn.CrossEntropyLoss(reduction='sum') # TODO: IGNASI: puede que haya losses mejores

    optimizer_class = torch.optim.Adam
    optimizer_params = dict(lr=lr, betas=(0.9, 0.999), weight_decay=0)
    optimizer = build_optimizer(optimizer_class, model, **optimizer_params)
    
    wandb_logger = WandbLogger(PROJECT_NAME, experiment_name, ENTITY)
    wandb_logger.watch_model(model, log="all", log_freq=80)
    weights_summary = "hand-picked" if isinstance(weights, (torch.Tensor, np.ndarray, list, tuple)) else weights
    hyperparameters = dict(embedding_dim=embedding_dim, weights=weights_summary, vector=vector, train_weights=train_weights, num_epochs=num_epochs, batch_size=batch_size, lr=lr)
    hyperparameters['optim_type'] = type(optimizer)
    wandb_logger.summarize(hyperparameters)

    train_accuracy = LocalLogger()
    wiki_accuracy = LocalLogger()
    valid_accuracy = LocalLogger()
    ############################################################################################

    #######################################   TRAINING   #######################################
    for epoch in range(num_epochs):
        acc, loss = train(model, criterion, optimizer, data[0][0], data[0][1], batch_size, device, train_accuracy, log=True)
        wandb_logger.log_epoch(train_accuracy.get_last_epoch_log(prefix="train_"), step=epoch)
        print(f'| epoch {epoch:03d} | train accuracy={acc:.1f}%, train loss={loss:.2f}')
        
        acc, loss = validate(model, criterion, data[1][0], data[1][1], batch_size, device, wiki_accuracy)
        wandb_logger.log_epoch(wiki_accuracy.get_last_epoch_log(prefix="wiki_"), step=epoch)
        print(f'| epoch {epoch:03d} | valid accuracy={acc:.1f}%, valid loss={loss:.2f} (wikipedia)')
        
        acc, loss = validate(model, criterion, valid_x, valid_y, batch_size, device, valid_accuracy)
        wandb_logger.log_epoch(valid_accuracy.get_last_epoch_log(prefix="valid_"), step=epoch)
        print(f'| epoch {epoch:03d} | valid accuracy={acc:.1f}%, valid loss={loss:.2f} (El Periódico)')

        # Save model
        torch.save(model.state_dict(), modelname)
        wandb_logger.upload_model(modelname, aliases=[f'epoch_{epoch}'], wait=(epoch==(num_epochs-1)))
        
        wandb_logger.log_epoch({'epoch' : epoch}, step=epoch, commit=True)

    ############################################################################################

    ####################################### POSTTRAINING #######################################
    best_epoch = valid_accuracy.best_epochs()[0] # TODO: IGNASI: deberia haber metricas mejores que accuracy, puede que decidir combinando resultados de wiki y periodico sea mejor.
    wandb_logger.update_model(f'epoch_{best_epoch}', ['best'])
    wandb_logger.summarize(train_accuracy.get_one_epoch_log(best_epoch, prefix="train_"))
    wandb_logger.summarize(wiki_accuracy.get_one_epoch_log(best_epoch, prefix="wiki_"))
    wandb_logger.summarize(valid_accuracy.get_one_epoch_log(best_epoch, prefix="valid_"))

    model_path = wandb_logger.download_model(os.path.basename(modelname), os.path.dirname(modelname), alias='best')
    model.load_state_dict(torch.load(model_path))

    wandb_logger.summarize(dict(best_weights=str(model.weights.data.view(-1))))

    ############################################################################################

    #######################################  SUBMISSION  #######################################
    # Submission generation
    y_pred = validate(model, None, test_x, None, batch_size, device)
    y_token = [vocab.idx2token[index] for index in y_pred]

    submission = pd.DataFrame({'id':valid_x_df['id'], 'token': y_token}, columns=['id', 'token'])
    print(submission.head())
    submission.to_csv(f'{OUTPUTS_ROOT}/submission.csv', index=False)

    wandb_logger.upload_submission(f'{OUTPUTS_ROOT}/submission.csv')
    ############################################################################################


if __name__ == "__main__":

    (experiment_name, weights, vector, train_weights, embedding_dim, shared_embedding, batch_size, fract, fract_dataset, epochs, lr, torch_seed, random_seed, numpy_seed) = parse_args(sys.argv[1:])

    params.embedding_dim = embedding_dim
    params.batch_size = batch_size
    params.epochs = epochs

    torch.manual_seed(torch_seed)
    random.seed(random_seed)
    np.random.seed(numpy_seed)

    # Create working dir
    #pathlib.Path(WORKING_ROOT).mkdir(parents=True, exist_ok=True)
    pathlib.Path(MODELS_ROOT).mkdir(parents=True, exist_ok=True)
    pathlib.Path(OUTPUTS_ROOT).mkdir(parents=True, exist_ok=True)

    # Select device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("WARNING: Training without GPU can be very slow!")

    main(params.window_size, params.embedding_dim,  weights, vector, train_weights, shared_embedding, params.epochs, params.batch_size, fract, fract_dataset, lr, params.preprocessed, params.modelname, experiment_name, device)

# TODO: Task 2 of assigment (study of the datasets and submission files) in another program
