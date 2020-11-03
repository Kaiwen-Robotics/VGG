"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

import torch
from torch import nn
# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.

  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch

  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  correct = 0
  for i in range(len(targets)):
      if(predictions[i] == targets[i]):
          correct += 1
  accuracy = correct/len(targets)
  #raise NotImplementedError
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model.

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)
  epoch_num = 3
  ########################
  # PUT YOUR CODE HERE  #
  #######################
  model = ConvNet(3, 10)

  #Obtain Dataset
  train_dataset = cifar10_utils.get_cifar10()['train']
  val_dataset = cifar10_utils.get_cifar10()['validation']
  test_dataset = cifar10_utils.get_cifar10()['test']

  train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE_DEFAULT,
        drop_last = True)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters())
  for epoch in range(epoch_num):
      model.train()
      losses = []
      accs = []
      for i_iter in range(int(train_dataset.num_examples/BATCH_SIZE_DEFAULT)):
          images, labels = train_dataset.next_batch(BATCH_SIZE_DEFAULT)
          images, labels = torch.tensor(images), torch.tensor(labels, dtype = torch.long)
          pred = model(images)
          labels = torch.argmax(labels, dim=1)
          loss = criterion(pred, labels)
          pred_result = torch.argmax(pred, dim=1)
          acc = accuracy(pred_result, labels)
          accs.append(acc)
          model.zero_grad()
          loss.backward()
          optimizer.step()
          losses.append(loss)
          if i_iter % 100 ==0:
              msg = 'Epoch:[{}/{}] Iter: [{}/{}], Loss: {: .6f}, ACC:[{: .6f}]'.format(epoch, epoch_num, i_iter, int(train_dataset.num_examples/BATCH_SIZE_DEFAULT), sum(losses)/len(losses), sum(accs)/len(accs))
              print(msg)
              with open('./log.txt', 'a') as f:
                  f.write(msg)
                  f.write('\n')
          msg_epoch = '--------Epoch: [{}/{}], Loss: {: .6f}, ACC:[{: .6f}]-------'.format(epoch, epoch_num, sum(losses)/len(losses), sum(accs)/len(accs))
          print(msg_epoch)
          with open('./log.txt', 'a') as f:
              f.write(msg)
              f.write('\n')
  #raise NotImplementedError
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
