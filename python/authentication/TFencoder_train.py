import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import tqdm
import math
from sklearn import metrics
import os
import argparse
import logging

import dataset
import TF_encoder as encoder


def compute_eer(label, pred, positive_label=1):
   # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
   fpr, tpr, threshold = metrics.roc_curve(y_true=label, y_score=pred, pos_label=positive_label)
   fnr = 1 - tpr

   # the threshold of fnr == fpr
   eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

   # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
   eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
   eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

   # return the mean of eer from fpr and from fnr
   eer = (eer_1 + eer_2) / 2
   return eer


def get_arguments():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument(
        "--train_in", type=str, metavar='', help="path of training data (.npy) files",
        default="../../ball_throwing_data/Vive1.npy"
    )
    parser.add_argument(
        "--test_in", type=str, metavar='', help="path of testing data (.npy) files",
        default="../../ball_throwing_data/Vive2.npy"
    )
    parser.add_argument(
        "-out_dir", type=str, metavar='', help="path to save output log file and params of the best model",
        default="../../classification_output/TF_encoder"
    )
    parser.add_argument(
        "--userID", "-u", type=int, metavar='', help="Train a network for a specific user",
        default=0
    )
    parser.add_argument(
        "--ws", type=int, metavar='', help="number of timestamps in a window",
        default=50
    )
    parser.add_argument(
        "--max_epoch", type=int, metavar='', help="max number of epoch",
        default=200
    )
    parser.add_argument(
        "--batch_size", type=int, metavar='', help=" ",
        default=64
    )
    parser.add_argument(
        "--lr", type=float, metavar='', help=" ",
        default=0.0001
    )
    parser.add_argument(
        "--optimizer", type=str, metavar='', help=" ",
        default="Adam", choices=['SGD', 'Adam']
    )
    parser.add_argument(
        "--checkpoint", type=str, metavar='', help="Saved model and optimizer params",
        default=None
    )
    parser.add_argument(
        "--decay_rate", type=float, metavar='', help='decay rate',
        default=1e-4,
    )
    parser.add_argument(
        "--log", "-l", action="store_true", help='save log files',
    )
    parser.add_argument(
        "--gpu", "-g", default=0,
    )
    
    return parser.parse_args()


if __name__ == "__main__":

    args = get_arguments()

    with torch.cuda.device('cuda:{}'.format(args.gpu)):
        torch.cuda.empty_cache()

    logger = logging.getLogger("TF_user{}_len{}".format(str(args.userID).zfill(2), args.ws))
    logger.setLevel(logging.DEBUG)
    fmt = "%(asctime)s [%(name)s] %(levelname)s>>> %(message)s"
    formatter = logging.Formatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if args.log:
        log_out_to = os.path.join(args.out_dir, "log", str(args.userID).zfill(2))
        if not os.path.exists(log_out_to):
            os.makedirs(log_out_to)

        fh = logging.FileHandler(os.path.join(
                    log_out_to,
                    "train_TF_{}_userID_{}_len_{}.log".format(
                        args.train_in.split("/")[-1].split(".")[0], 
                        str(args.userID).zfill(2), 
                        str(args.ws).zfill(2),
                        )
                    ), 
                'w'
                )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.debug("Model:           {}".format("Vanilla Transformer Encoder"))
    logger.debug("Training data:   {}".format(args.train_in))
    logger.debug("Testing data:    {}".format(args.test_in))
    logger.debug("data length:     {}".format(args.ws))
    logger.debug("Number of epoch: {}".format(args.max_epoch))
    logger.debug("Batch size:      {}".format(args.batch_size))
    logger.debug("Optimizer:       {}".format(args.optimizer))
    logger.debug("Learning rate:   {}".format(args.lr))

    device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    logger.debug("Using device:    {}\n".format(device))

    start_epoch = 0
    best_test_acc = -math.inf
    best_test_acc_epoch = 0
    best_eer = math.inf
    best_eer_epoch = 0

    train_data_path = args.train_in
    test_data_path = args.test_in

    train_data = dataset.VIVE(train_data_path, args.userID, window_size=args.ws, stride=1)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    test_data = dataset.VIVE(test_data_path, args.userID, window_size=args.ws, stride=1)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    transformerEncoder = encoder.Encoder(seq_len=args.ws).to(device)

    criterion2 = nn.BCELoss().to(device)

    if args.optimizer == "SGD":
        optimizer = optim.SGD(transformerEncoder.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(
            transformerEncoder.parameters(), 
            lr = args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )

    # check if there is a previous process not finished
    if not args.checkpoint == None:
        checkpoint = torch.load(args.checkpoint)
        logger.info("Using params from a pretrained model...")
        
        transformerEncoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    for epoch in tqdm.tqdm(range(start_epoch, args.max_epoch)):
        # Train    
        transformerEncoder = transformerEncoder.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels2 = torch.eye(2)[labels.long(), :].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs, _ = transformerEncoder(inputs)

            m = nn.Sigmoid()
            loss = criterion2(m(outputs), labels2)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        logger.info("Epoch {}/{} Training loss   : {}".format(epoch + 1, args.max_epoch, running_loss))

        # evaluation
        batch_loss = 0
        correct_t = 0
        total_t = 0
        with torch.no_grad():
            transformerEncoder.eval()
            all_labels_t = []
            all_preds_t = []
            for i_t, data_t in enumerate(test_loader, 0):
                inputs_t, labels_t = data_t
                all_labels_t.append(labels_t.tolist())
                
                inputs_t = inputs_t.to(device)
                labels_t1 = labels_t.to(device)
                labels_t2 = torch.eye(2)[labels_t.long(), :].to(device)

                outputs_t, _ = transformerEncoder(inputs_t)
                m = nn.Sigmoid()

                loss_t = criterion2(m(outputs_t), labels_t2)
                batch_loss += loss_t.item()
                _, pred_t = torch.max(outputs_t, dim=1)
                all_preds_t.append(pred_t.cpu().tolist())

                correct_t += torch.sum(pred_t==labels_t1).item()
                total_t += labels_t.size(0)

            # compute equal error rate
            flatten_list = lambda y:[x for a in y for x in flatten_list(a)] if type(y) is list else [y]
            all_labels_t = flatten_list(all_labels_t)
            all_preds_t = flatten_list(all_preds_t)

            eer = compute_eer(all_labels_t, all_preds_t)

            check_out_to = os.path.join(args.out_dir, "checkpoints", str(args.userID).zfill(2))
            if not os.path.exists(check_out_to):
                os.makedirs(check_out_to)

            if best_test_acc < 100 * correct_t / total_t:
                best_test_acc = 100 * correct_t / total_t
                best_test_acc_epoch = epoch + 1
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": transformerEncoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    },
                    os.path.join(
                        check_out_to,
                        "TF_best_model_userID_{}_len_{}.pth".format(str(args.userID).zfill(2), str(args.ws).zfill(2))
                    )
                    
                )
                logger.debug("Save model in epoch {} with testing ACC {}.".format(epoch+1, 100 * correct_t / total_t))

            if eer < best_eer:
                best_eer = eer
                best_eer_epoch = epoch + 1
        

            logger.debug("Epoch {}/{} Evaluation ACC  : {}".format(epoch + 1, args.max_epoch, 100 * correct_t / total_t))
            logger.debug("Epoch {}/{} Equal Error Rate: {}".format(epoch + 1, args.max_epoch, eer))
            logger.debug("Best Evaluation ACC  : {} at epoch {}".format(best_test_acc, best_test_acc_epoch))
            logger.debug("Best Equal Error Rate: {} at epoch {}".format(best_eer, best_eer_epoch))

    logger.info("--"*30)
    logger.info("Best Evaluation ACC  : {} at epoch {}".format(best_test_acc, best_test_acc_epoch))
    logger.info("Best Equal Error Rate: {} at epoch {}".format(best_eer, best_eer_epoch))
    logger.info("--"*30)
    print('Finished Training')