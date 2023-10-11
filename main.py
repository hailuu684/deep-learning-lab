from pytorch_lightning.loggers import WandbLogger

import lab02, lab03
import pytorch_lightning as pl
import torch
# Import wandb
import wandb


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


if __name__ == '__main__':
    main(lab='03')
