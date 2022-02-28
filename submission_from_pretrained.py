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


BATCH_SIZE = 1
SHARED_EMBEDDING = False
DOCTEXT = f"""
Usage:
  submission_from_pretrained --embedding_dim=<ed> --model_path=<mp> [--batch_size=<bs>] [--shared_embedding=<se>]
  submission_from_pretrained -h | --help


Options:
  --embedding_dim=<ed>          Int. Number of components of the embedding space of the model.
  --model_path=<mp>             System path to the model parameters (state_dict).
  --batch_size=<bs>             Int. Number of samples to be processed simultaneously (before applying a optimization step) [default: {BATCH_SIZE}].
  --shared_embedding=<se>       Bool. Set to True in order to use the embedding transposed on the linear layer weigths [default: {SHARED_EMBEDDING}].
"""


DATASET_VERSION = "ca-100"
COMPETITION_ROOT = "data/word_vectors"  # I downloaded it from https://www.kaggle.com/c/vectors3/data
DATASET_ROOT = f"data/text-preprocessing/data/{DATASET_VERSION}"
MODELS_ROOT = f"model_parameters/{DATASET_VERSION}"
OUTPUTS_ROOT = f"outputs/{DATASET_VERSION}"
DATASET_PREFIX = "ca.wiki"

params = SimpleNamespace(
    embedding_dim=EMBEDDING_DIM,
    window_size=7,
    batch_size=1000,
    epochs=4,
    preprocessed=f"{DATASET_ROOT}/{DATASET_PREFIX}",
    # working = f'{WORKING_ROOT}/{DATASET_PREFIX}',
    working=f"{OUTPUTS_ROOT}/{DATASET_PREFIX}",  # Not used anywhere
    # modelname = f'{WORKING_ROOT}/{DATASET_VERSION}.pt',
    modelname=f"{MODELS_ROOT}/{DATASET_VERSION}.pt",
    train=True,  # Not used anywhere
)


def build_periodico_dataset(vocab):
    # IGNSAI: Funcion un poco fea porque depende directamente de constantes o hard-wired text

    # 'El Periodico' validation dataset
    valid_x_df = pd.read_csv(f"{COMPETITION_ROOT}/x_valid.csv")
    tokens = valid_x_df.columns[1:]
    valid_x = valid_x_df[tokens].apply(vocab.get_index).to_numpy(dtype="int32")
    valid_y_df = pd.read_csv(f"{COMPETITION_ROOT}/y_valid.csv")
    valid_y = valid_y_df["token"].apply(vocab.get_index).to_numpy(dtype="int32")

    # 'El Periodico' test dataset
    valid_x_df = pd.read_csv(f"{COMPETITION_ROOT}/x_test.csv")
    test_x = valid_x_df[tokens].apply(vocab.get_index).to_numpy(dtype="int32")

    return valid_x, valid_y, valid_x_df, test_x

def validate(model, criterion, idata, target, batch_size, device, local_logger=None):
    model.eval()
    if local_logger is not None:
        local_logger.new_epoch()

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
                pred = torch.max(output, 1)[1].detach().to("cpu").numpy()
                y_pred.append(pred)

    if target is not None:
        local_logger.finish_epoch(VERBOSE=False)
        return local_logger["accuracy"][-1], local_logger["total_loss"][-1]

    else:
        return np.concatenate(y_pred)


if __name__ == "__main__":

    (window_size, embedding_dim, num_epochs, batch_size, preprocessed_path, modelname,) = (
        params.window_size,
        params.embedding_dim,
        params.epochs,
        params.batch_size,
        params.preprocessed,
        params.modelname,
    )

    opts = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    embedding_dim = int(opts['--embedding_dim'])
    batch_size = int(opts['--batch_size'])
    shared_embedding = True if opts['--shared_embedding'] == 'True' else False
    model_path = opts['--model_path']

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("WARNING: Training without GPU can be very slow!")

    model = CBOW(len(vocab), embedding_dim, num_context_words=window_size - 1, vector=True, shared_embedding=shared_embedding).to(device)
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
        )

    y_pred = validate(model, None, test_x, None, batch_size, device)
    y_token = [vocab.idx2token[index] for index in y_pred]

    submission = pd.DataFrame({'id':valid_x_df['id'], 'token': y_token}, columns=['id', 'token'])
    print(submission.head())
    submission.to_csv(f'{OUTPUTS_ROOT}/submission.csv', index=False)
