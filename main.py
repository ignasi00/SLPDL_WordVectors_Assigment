
from datetime import datetime
import numpy as np
import os
import pandas as pd
import pathlib
import torch
import torch.nn as nn
from types import SimpleNamespace

from data_loading import Vocabulary, batch_generator, load_preprocessed_dataset
from local_logger import LocalLogger
from model import CBOW


PROJECT_NAME = "SLPDL_WordVectors"
EXPERIMENT_NAME = datetime.now().strftime("%Y%m%d%H%M_trained-vector") # %Y%m%d%H%M_xxxx <- default, fixed, trained-scalar and trained-vector # TODO: en docopts, dependiendo de parametros un nombre u otro
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
    embedding_dim = 100,                                            # TODO: hyperparam optimization
    window_size = 7,
    batch_size = 1000,                                              # TODO: hyperparam optimization
    epochs = 4,                                                     # TODO: hyperparam optimization
    preprocessed = f'{DATASET_ROOT}/{DATASET_PREFIX}',
    #working = f'{WORKING_ROOT}/{DATASET_PREFIX}',
    working = f'{OUTPUTS_ROOT}/{DATASET_PREFIX}', # Not used anywhere
    #modelname = f'{WORKING_ROOT}/{DATASET_VERSION}.pt',
    modelname = f'{MODELS_ROOT}/{DATASET_VERSION}.pt',
    train = True
)


def train(model, criterion, optimizer, idata, target, batch_size, device, local_logger, log=False):
    model.train()
    local_logger.new_epoch()

    for X, y in batch_generator(idata, target, batch_size, shuffle=True):
        # Get input and target sequences from batch
        X = torch.tensor(X, dtype=torch.long, device=device)
        y = torch.tensor(y, dtype=torch.long, device=device)

        model.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # Training statistics
        local_logger.update_epoch_log(output, y, loss, VERBOSE=True)

    local_logger.finish_epoch(VERBOSE=log)
    return local_logger['accuracy'][-1], local_logger['total_loss'][-1]

def validate(model, criterion, idata, target, batch_size, device, local_logger=None):
    model.eval()
    if local_logger is not None : local_logger.new_epoch()

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

if __name__ == "__main__":

    # TODO: all main into main function with parameters and here call docopts from docopts_parser.py and call the function as needed, additionally tracking random seeds may be usful (docopts + wandb)

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

    vocab, data = load_preprocessed_dataset(params.preprocessed)

    # 'El Periodico' validation dataset
    valid_x_df = pd.read_csv(f'{COMPETITION_ROOT}/x_valid.csv')
    tokens = valid_x_df.columns[1:]
    valid_x = valid_x_df[tokens].apply(vocab.get_index).to_numpy(dtype='int32')
    valid_y_df = pd.read_csv(f'{COMPETITION_ROOT}/y_valid.csv')
    valid_y = valid_y_df['token'].apply(vocab.get_index).to_numpy(dtype='int32')

    # 'El Periodico' test dataset
    valid_x_df = pd.read_csv(f'{COMPETITION_ROOT}/x_test.csv')
    test_x = valid_x_df[tokens].apply(vocab.get_index).to_numpy(dtype='int32')

    model = CBOW(len(vocab), params.embedding_dim).to(device) # TODO: section D, modify model to study sharing Input/output embedings. Section E, modify model to improve the embeddings.
    print(model)
    for name, param in model.named_parameters():
        print(f'{name:20} {param.numel()} {list(param.shape)}')
    print(f'TOTAL{" " * 16}{sum(p.numel() for p in model.parameters())}')

    criterion = nn.CrossEntropyLoss(reduction='sum') # IGNASI: puede que haya losses mejores

    optimizer = torch.optim.Adam(model.parameters()) # TODO: hyperparam optimization (optimizer & learning_rate including not time/space constant learning_rates)

    wandb_logger = WandbLogger(PROJECT_NAME, EXPERIMENT_NAME, ENTITY)
    wandb_logger.watch_model(model)

    train_accuracy = LocalLogger()
    wiki_accuracy = LocalLogger()
    valid_accuracy = LocalLogger()
    for epoch in range(params.epochs):
        acc, loss = train(model, criterion, optimizer, data[0][0], data[0][1], params.batch_size, device, train_accuracy, log=True)
        wandb_logger.log_epoch(train_accuracy.get_last_epoch_log(prefix="train_"), step=epoch)
        print(f'| epoch {epoch:03d} | train accuracy={acc:.1f}%, train loss={loss:.2f}')
        
        acc, loss = validate(model, criterion, data[1][0], data[1][1], params.batch_size, device, wiki_accuracy)
        wandb_logger.log_epoch(train_accuracy.get_last_epoch_log(prefix="wiki_"), step=epoch)
        print(f'| epoch {epoch:03d} | valid accuracy={acc:.1f}%, valid loss={loss:.2f} (wikipedia)')
        
        acc, loss = validate(model, criterion, valid_x, valid_y, params.batch_size, device, valid_accuracy)
        wandb_logger.log_epoch(train_accuracy.get_last_epoch_log(prefix="valid_"), step=epoch)
        print(f'| epoch {epoch:03d} | valid accuracy={acc:.1f}%, valid loss={loss:.2f} (El Periódico)')

        # Save model
        torch.save(model.state_dict(), params.modelname)

        wandb_logger.upload_model(params.modelname, aliases=[f'epoch_{epoch}'])

    # TODO: Process local_logger in order to set a "best" model (to a given epoch) tag and summarize the experiment in wandb
    #xxxx = algo(some_local_loggers) # IGNASI: deberia haber metricas mejores que accuracy
    #wandb_logger.update_model(f'epoch_{xxxx}', ['best'])
    #wandb_logger.summarize <- diccionario de cosas a guardar, por ejemplo, estadisticas de todo de la mejor epoch o parametros de entrada
        
    # TODO: Get the "best" epoch model from wandb in order to generate the submission
    #wandb_logger.download_model(os.paths.basename(params.modelname), os.path.dirname(params.modelname), alias=['best'])

    # Submission generation
    y_pred = validate(model, None, test_x, None, params.batch_size, device)
    y_token = [vocab.idx2token[index] for index in y_pred]

    submission = pd.DataFrame({'id':valid_x_df['id'], 'token': y_token}, columns=['id', 'token'])
    print(submission.head())
    submission.to_csv(f'{OUTPUTS_ROOT}/submission.csv', index=False)

    # TODO: Save submission on wandb artifact
    wandb_logger.upload_submission(f'{OUTPUTS_ROOT}/submission.csv')

# TODO: Task 2 of assigment (study of the datasets and submission files) in another program
