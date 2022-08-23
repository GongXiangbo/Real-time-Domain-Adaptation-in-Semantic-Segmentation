import os
import torch
import numpy as np


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_onehot_vector(args, labels):
    labels = labels.long().unsqueeze(1)
    labels[labels == 255] = args.num_classes
    bs, _, h, w = labels.size()
    nc = args.num_classes + 1
    input_label = torch.FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, labels, 1.0)[0][:args.num_classes].unsqueeze(0)
    return input_semantics


def fast_hist(a, b, n):
    '''
    a and b are predict and mask respectively
    n is the number of classes
    '''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
      image: The one-hot format image

    # Returns
      A 2D array with the same width and height as the input, but
      with a depth size of 1, where each pixel value is the classified
      class key.
    """
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,1])

    # for i in range(0, w):
    #     for j in range(0, h):
    #         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
    #         x[i, j] = index
    image = image.permute(1, 2, 0)
    x = torch.argmax(image, dim=-1)
    return x


def compute_global_accuracy(pred, label):
    pred = pred.flatten()
    label = label.flatten()
    total = len(label)
    count = 0.0
    for i in range(total):
      if pred[i] == label[i]:
        count = count + 1.0
    return float(count) / float(total)
