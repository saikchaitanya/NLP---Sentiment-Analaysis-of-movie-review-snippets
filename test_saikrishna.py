# some housekeeping libraries
from absl import app, flags, logging
#import sh 
#import sh doesn't work for windows. Sh package only works on Unix-like systems (OSX, Linux, etc). (https://github.com/kivy/python-for-android/issues/1721)
import pbs as sh


# import pytorch
import torch as th

# import pytorch_lightning
# this library simplifies training significantly
# familiarize yourself with the principals first: https://pytorch-lightning.readthedocs.io/en/latest/
import pytorch_lightning as pl

# hugging face libraries
# https://github.com/huggingface/nlp/
import nlp
# https://github.com/huggingface/transformers/
import transformers
import sklearn as sk

# parameters for the network
# these have been tested and the network trains appropriately when implemented correctly
# use FLAGS.debug=True to test your network (it will not run an entire training epoch for this) see: https://pytorch-lightning.readthedocs.io/en/latest/debugging.html#fast-dev-run
flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_string('dataset', 'rotten_tomatoes', '')
flags.DEFINE_integer('epochs', 10, '')
flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_integer('seq_length', 20, '')
flags.DEFINE_float('learning_rate', 1e-4, '')
flags.DEFINE_float('momentum', .9, '')
flags.DEFINE_string('model', 'bert-base-uncased', '')
# you might need to change this depending on your machine
flags.DEFINE_integer('num_workers', 0, '')

FLAGS = flags.FLAGS

# clears the logs for you
#sh.rm('-r', '-f', 'logs')
#sh.mkdir('logs')


# define the module
import torch.nn as nn
from transformers import BertModel
import torch

#from pytorch_lightning.metrics.classification import Accuracy
# all functions and parts of the code to be implemented are marked with *********Implement**************
class RTSentimentClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        #model = BertModel.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained(FLAGS.model)
        self.linear = nn.Linear(768, 1) # This is to define the linear layer after each pooled output
        self.sigmoid = nn.Sigmoid() # To use as the activation function
        self.loss_func = nn.BCELoss() # As we are working with only 2 classes, use binary cross entropy loss

        # *********Implement**************
        # initialize your model here and make use of the pre-trained BERT model defined in FLAGS.model
        # further define your loss function here. leverage the pytorch library for this purpose
        
    # this function prepares the data for you and uses the tokenizer from the pretrained model
    def prepare_data(self):
      
        #tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer = transformers.BertTokenizer.from_pretrained(FLAGS.model)
        '''
        print(tokenizer.tokenize('Hello WORLD how ARE yoU?'))
        print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('Hello WORLD how ARE yoU?')))
        '''
        

        def _tokenize(x):
            x['input_ids'] = tokenizer.batch_encode_plus(
                    x['text'],
                    #max_length = 20,
                    max_length=FLAGS.seq_length, 
                    pad_to_max_length=True)['input_ids']
            return x

        def _prepare_ds(split):
            #ds = nlp.load_dataset('rotten_tomatoes', split=f'{split}')
            ds = nlp.load_dataset(FLAGS.dataset, split = f'{split}')
            ds = ds.map(_tokenize, batched=True)
            ds.set_format(type='torch', columns=['input_ids', 'label'])
            return ds
        #train_ds, validation_ds, test_ds = map(_prepare_ds, ('train', 'validation', 'test'))  
        self.train_ds, self.validation_ds, self.test_ds = map(_prepare_ds, ('train', 'validation', 'test'))
        
    # *********Implement**************
    # this function implements the forward step in your network
    def forward(self, input_ids):
      _, pooled_output = self.model(input_ids)
      linear_output = self.linear(pooled_output)
      proba = self.sigmoid(linear_output) # Take the probability after sigmoid activation
      #output = self.out(output)
      return proba
      
    
    # *********Implement**************
    # this function defines the training step
    def training_step(self, batch, batch_idx):
      #batch
      label = batch['label']
      input_ids = batch['input_ids']     
      #generated output
      y_hat = self(input_ids).float()
      label = label.float()
      #y_hat = y_hat.view(8)     
      #loss
      loss = self.loss_func(y_hat, label) # Converted both the predicted and target variable to same data type      
      #logs
      tensorboard_logs = {'train_loss': loss}     
      
      return {'loss':loss, 'log':tensorboard_logs}
      
    # *********Implement**************
    # this function defines the validation step
    def validation_step(self, batch, batch_idx):
      # batch
      label = batch['label']
      input_ids = batch['input_ids']     
      # fwd
      y_hat = self(input_ids).float()
      label = label.float()
      #y_hat = y_hat.view(8)
      # loss
      loss = self.loss_func(y_hat, label) # Converted both the predicted and target variable to same data type     
      # acc
      a, y_hat = torch.max(y_hat, dim=1)
      val_acc = sk.metrics.accuracy_score(y_hat.cpu(), label.cpu())
      #val_f1 = sk.metrics.f1_score(y_hat.cpu(), label.cpu())
      val_cm = sk.metrics.confusion_matrix(y_hat.cpu(), label.cpu())
      val_acc = torch.tensor(val_acc)
      #val_f1 = torch.tensor(val_f1)
      val_cm = torch.tensor(val_cm)
      return {'val_loss': loss, 'val_acc': val_acc, 'val_cm' : val_cm}
        
    # *********Implement**************
    # this function concludes the validation loop
    def validation_epoch_end(self, outputs):
      avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean() # Taking mean of val_loss
      avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean() # Taking mean of val_acc
      #avg_val_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()
      tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}
      #self.logger.experiment.log_metrics('val_loss', avg_loss.detach().cpu().numpy(), step=self.global_step, epoch=self.current_epoch, include_context=True)
      return {'val_loss': avg_loss, 'progress_bar': tensorboard_logs}
      
    # *********Implement**************
    # this function defines the test step
    def test_step(self, batch, batch_idx):
      label = batch['label']
      input_ids = batch['input_ids']
      y_hat = self(input_ids).float()
      label = label.float()
      #y_hat = y_hat.view(8)
      a, y_hat = torch.max(y_hat, dim=1)
      test_acc = sk.metrics.accuracy_score(y_hat.cpu(), label.cpu()) # Converted both the predicted and target variable to same data type
      #test_f1 = sk.metrics.f1_score(y_hat.cpu(), label.cpu())  
      return {'test_acc': torch.tensor(test_acc)}
        
    # *********Implement**************
    # this function concludes the test loop
    def test_epoch_end(self, outputs):
      avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
      #avg_test_f1 = torch.stack([x['test_f1'] for x in outputs]).mean()
      tensorboard_logs = {'avg_test_acc': avg_test_acc}
      return {'avg_test_acc': avg_test_acc, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}
        
    # this function defines the training data for you
    def train_dataloader(self):
        return th.utils.data.DataLoader(
                self.train_ds,
                batch_size=FLAGS.batch_size,
                drop_last=True,
                shuffle=True,
                num_workers=FLAGS.num_workers,
                )

    # this function defines the validation data for you
    def val_dataloader(self):
        return th.utils.data.DataLoader(
                self.validation_ds,
                batch_size=FLAGS.batch_size,
                drop_last=False,
                shuffle=False,
                num_workers=FLAGS.num_workers,
                )
    
    # this function defines the test data for you
    def test_dataloader(self):
        return th.utils.data.DataLoader(
                self.test_ds,
                batch_size=FLAGS.batch_size,
                drop_last=False,
                shuffle=False,
                num_workers=FLAGS.num_workers,
                )
    
    
    # *********Implement**************
    # here you define the appropriate optimizer (use SGD the only one tested for this)
    # use the pytorch library for this
    # make sure to use the parameters defined in FLAGS
    #params = list(model.named_parameters())
    def configure_optimizers(self):
      import torch.optim as optim
      optimizer = optim.SGD(params = self.model.parameters(), lr = FLAGS.learning_rate, momentum = FLAGS.momentum) # Selecting SGD optimizer with given params
      return optimizer

def main(_):
    # *********Implement**************
    # initialize your model and trainer here
    # further fit the model and don't forget to run the test; pytorch lightning does not automatically do that for you!
    bert_fin_model = RTSentimentClassifier()
    # most basic trainer, uses good defaults 
    logger = pl.loggers.TensorBoardLogger(save_dir='experiments', version=10)
    trainer = pl.Trainer(profiler=True, fast_dev_run=FLAGS.debug, logger = logger, show_progress_bar= True , max_epochs=FLAGS.epochs)     # defaut trainer of lightning module. 
    trainer.fit(bert_fin_model)
    
if __name__ == '__main__':
    app.run(main)