import lab02
import pytorch_lightning as pl


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


if __name__ == '__main__':
    main(lab='02')
