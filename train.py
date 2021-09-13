import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from cassava_lit_mods import CassavaLitMod, CassavaDataMod

def main():
    logger = WandbLogger(project='cassava-leaf-disease-classifier', log_model=False, config=wandb.config)
    config = logger.experiment.config
    callbacks = [
        #EarlyStopping('val_loss', patience=5),
        LearningRateMonitor(logging_interval='step'),
        ]
    trainer = Trainer(max_epochs=config['epochs'], logger=logger, callbacks=callbacks, gpus=1)
    model = CassavaLitMod(lr=config['learning_rate'], hidden_size=config['hidden_size'], wd=config['weight_decay'])
    data = CassavaDataMod(batch_size=config['batch_size'])
    trainer.fit(model, data)

if __name__ == '__main__':
    main()