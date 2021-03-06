
import wandb
import shutil


class WandbLogger():

    def __init__(self, project_name, experiment_name, entity):
        # wandb.init creates (and return) a wandb.Run object available on wandb.run
        # only 1 run should exist at the same script at the same time...
        wandb.init(project=f"{project_name}", entity=entity)

        self.entity = entity
        self.project_name = project_name
        self.experiment_name = experiment_name

    def watch_model(self, model, log="all", log_freq=1000):
        # watch a given run
        wandb.watch(model, log=log, log_freq=log_freq)

    def log_batch(self, dict_data, step=None, commit=None):
        # TODO: externally manage different indexes like batch_idx, #examples, etc
        wandb.log(dict_data, step=step, commit=commit)

    def log_epoch(self, dict_data, step=None, commit=None, prefix=None):
        # TODO: externally manage different indexes like epoch_idx; in fact, this is exactly equal to log_batch
        if prefix : dict_data = { f'{prefix}{k}' : v for k, v in dict_data.items()}
        wandb.log(dict_data, step=step, commit=commit)

    def upload_model(self, model_file, aliases=None, wait=True):
        model_io = wandb.Artifact(self.experiment_name, type="model_parameters")
        model_io.add_file(model_file)

        aliases = aliases or []
        aliases = ['latest'] + aliases

        wandb.log_artifact(model_io, aliases=aliases)
        if wait: model_io.wait()

    def update_model(self, search_alia, new_alias_list):
        model_io = wandb.run.use_artifact(f'{self.entity}/{self.project_name}/{self.experiment_name}:{search_alia}', type="model_parameters")
        for alia in new_alias_list:
            model_io.aliases.append(alia)
        model_io.save()

    def upload_submission(self, submission_file, aliases=None, wait=False):
        submission_io = wandb.Artifact(f'{self.experiment_name}_submission', type="submissions")
        submission_io.add_file(submission_file)

        aliases = aliases or []
        aliases = ['latest'] + aliases

        wandb.log_artifact(submission_io, aliases=aliases)
        if wait: submission_io.wait()
    
    def summarize(self, dict_data):
        # log on a given run single final values like best scores (overwritting), configs, best epoch, etc
        for key, value in dict_data.items():
            wandb.run.summary[key] = value

    def download_model(self, model_filename, output_dir, alias=None):
        alias = alias or "latest"
        # Query W&B for an artifact and mark it as input to this run
        artifact = wandb.run.use_artifact(f'{self.entity}/{self.project_name}/{self.experiment_name}:{alias}', type="model_parameters")

        # Download the artifact's contents
        artifact_dir = artifact.download()

        # TODO: ??Move it where it should be?
        shutil.move(f"{artifact_dir}/{model_filename}", f"{output_dir}/{model_filename}")

        return f"{output_dir}/{model_filename}"
