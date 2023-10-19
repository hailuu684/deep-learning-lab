import pytorch_lightning as pl
# Import wandb
import wandb
from pytorch_lightning.loggers import WandbLogger

import lab02
import lab03
import lab04


def main(lab):
    if lab == '02':
        # instantiate classes
        dm = lab02.CIFAR10DataModule(batch_size=32, data_dir='/home/luu/DL_lab')
        dm.prepare_data()
        dm.setup()
        model = lab02.CIFARLitModel((3, 32, 32), dm.num_classes)

        # Initialize Callbacks
        checkpoint_callback = pl.callbacks.ModelCheckpoint()
        early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_acc", patience=3, verbose=False, mode="max")
        trainer = pl.Trainer(max_epochs=10,
                             callbacks=[checkpoint_callback, early_stop_callback]
                             )

        # Train the model
        trainer.fit(model, dm)

        # Evaluate the model
        trainer.test(dataloaders=dm.test_dataloader())

    elif lab == '03':
        key = '1bed216d1f9c32afa692155d2e0911cd750f41dd'
        wandb.login(key=key)

        # instantiate classes
        dm = lab02.CIFAR10DataModule(batch_size=32, data_dir='/home/luu/DL_lab')
        dm.prepare_data()
        dm.setup()
        model = lab03.CIFARLitModel((3, 32, 32), dm.num_classes)

        from models import resnet
        resnet_model = resnet.ResNet(resnet.ResNetBlock, [2, 2, 2, 2])
        lightning_resnet_model = lab03.ResNetLightning(resnet_model)

        # start a new wandb run to track this script
        wandb_logger = WandbLogger(project="lab-03")

        callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath="checkpoints",
                every_n_train_steps=100,
            ),
        ]

        trainer = pl.Trainer(
            max_epochs=100,
            logger=wandb_logger,
            callbacks=callbacks
        )

        # Train the model
        trainer.fit(model, dm)

        # Evaluate the model
        trainer.test(dataloaders=dm.test_dataloader())

    elif lab == '04':
        key = '1bed216d1f9c32afa692155d2e0911cd750f41dd'
        wandb.login(key=key)

        # instantiate classes
        dm = lab02.CIFAR10DataModule(batch_size=32, data_dir='/home/luu/DL_lab')
        dm.prepare_data()
        dm.setup()
        model = lab03.CIFARLitModel((3, 32, 32), dm.num_classes)

        # start a new wandb run to track this script
        wandb_logger = WandbLogger(entity='deep-learning-lab-msc')

        callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath="checkpoints",
                every_n_train_steps=100,
            ),
        ]

        trainer = pl.Trainer(
            max_epochs=100,
            logger=wandb_logger,
            callbacks=callbacks
        )

        # Train the model
        wandb.require(experiment="service")
        trainer.fit(model, dm)

        # Evaluate the model
        trainer.test(dataloaders=dm.test_dataloader())


def get_csv_results():
    import pandas as pd
    api = wandb.Api()
    entity, project = "deep-learning-lab-msc", "sweep-lab04"
    runs = api.runs(entity + "/" + project)

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains output keys/values for
        # metrics such as accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items()
             if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
    })

    runs_df.to_csv("./project.csv")
    print("saved to csv file")


if __name__ == '__main__':
    # main(lab='03')

    # #Sweep
    # sweep_config = {
    #     'method': 'random',
    #     'name': 'first_sweep',
    #     'metric': {
    #         'goal': 'minimize',
    #         'name': 'loss'
    #     },
    #     'parameters': {
    #         'n_hidden': {'values': [2, 3, 5, 10]},
    #         'lr': {'max': 1.0, 'min': 0.0001, 'distribution': 'uniform'},
    #         'batch_size': {
    #             # integers between 32 and 256
    #             # with evenly-distributed logarithms
    #             'distribution': 'q_log_uniform_values',
    #             'q': 8,
    #             'min': 32,
    #             'max': 256,
    #         },
    #         'epochs': {'values': [5, 10, 15, 30]},
    #
    #         'optimizer': {
    #             'values': ['adam', 'sgd']
    #         },
    #         'fc_layer_size': {
    #             'values': [128, 256, 512]
    #         },
    #         'dropout': {
    #             'values': [0.3, 0.4, 0.5, 0.7]
    #         }
    #     },
    #     'optimizer': {
    #         'values': ['adam', 'sgd', 'adamw', 'nadam', 'rprop']
    #     }
    # }
    #
    # sweep_id = wandb.sweep(sweep_config, project="sweep-lab04")
    #
    # wandb.agent(sweep_id=sweep_id, function=lab04.train, count=20)

    get_csv_results()
