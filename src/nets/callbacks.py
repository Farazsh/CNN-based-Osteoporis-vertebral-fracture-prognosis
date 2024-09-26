import torch
import torch.nn.functional as F
from pytorch_lightning import Callback


class LogPredictionSamplesCallback(Callback):

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        inputs, targets = batch
        # predictions = self(inputs).type(torch.FloatTensor)  # needs to be casted to a float tensor for cce
        targets = targets.squeeze().type(torch.LongTensor)
        validation_loss = F.cross_entropy(outputs, targets)
        # trainer.logger.log_metrics({'val epoch end loss':validation_loss.})