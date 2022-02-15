
import wandb
import shutil


class WandbLogger():

    def __init__(self, project_name, experiment_name, entity):
        # wandb.init creates (and return) a wandb.Run object available on wandb.run
        # only 1 run should exist at the same script at the same time...
        wandb.init(project=f"{project_name}", entity=entity)

        self.experiment_name = experiment_name

    def watch_model(self, model, log="all", log_freq=1000):
        # watch a given run
        wandb.watch(model, log=log, log_freq=log_freq)

    def log_batch(self, dict_data, step=None, commit=None):
        # TODO: externally manage different indexes like batch_idx, #examples, etc
        wandb.log(dict_data, step=step, commit=commit)

    def log_epoch(self, dict_data, step=None, commit=None):
        # TODO: externally manage different indexes like epoch_idx; in fact, this is exactly equal to log_batch
        wandb.log(dict_data, step=step, commit=commit)

    def upload_model(self, model_file, aliases=None):
        model_io = wandb.Artifact(self.experiment_name, type="model_parameters")
        model_io.add_file(model_file)

        aliases = aliases or []
        aliases = ['latest'] + aliases

        wandb.log_artifact(model_io, aliases=aliases)

    def upload_submission(self, submission_file, aliases=None):
        submission_io = wandb.Artifact(self.experiment_name, type="submissions")
        submission_io.add_file(model_file)

        aliases = aliases or []
        aliases = ['latest'] + aliases

        wandb.log_artifact(submission_io, aliases=aliases)
    
    def summarize(self, dict_data):
        # log on a given run single final values like best scores (overwritting), configs, best epoch, etc
        for key, value in dict_data.items():
            wandb.run.summary[key] = value

    def download_model(self, model_filename, output_dir, alias=None):
        alias = alias or "latest"
        # Query W&B for an artifact and mark it as input to this run
        artifact = run.use_artifact(f'{self.experiment_name}:{alias}')

        # Download the artifact's contents
        artifact_dir = artifact.download()

        # TODO: ¿Move it where it should be?
        shutil.move(f"{artifact_dir}/{model_filename}", f"{output_dir}/{model_filename}")

        return f"{output_dir}/{model_filename}"
