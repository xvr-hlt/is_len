def test_datamodule(datamodule):
    for i in (datamodule.train_dataloader(), datamodule.val_dataloader()):
        assert next(iter(i))
