import os
import numpy as np
import tqdm
import argparse
import logging
from sklearn import metrics
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from data import Fore_Auth
from models.main_model import Fore_Auth_Model


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


def get_argumets():
    """
        Parse arguments from command line
    """
    parser = argparse.ArgumentParser(description='Forecasting for Authentication')
    parser.add_argument('--data_root', type=str, required=False, default='../../ball_throwing_data',help='')
    parser.add_argument('--out_root', type=str, required=False, default="../../Fore_Auth_output",help='')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--user_id', type=int, default=0, help='user ID for Vive dataset')
    parser.add_argument('--learning_rate', type=int, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--max_epoch', type=int, default=200, help='')

    parser.add_argument('--seq_len', type=int, default=20, help='input sequence length of Informer encoder')
    parser.add_argument('--label_len', type=int, default=10, help='start token length of Informer decoder')
    parser.add_argument('--pred_len', type=int, default=5, help='prediction sequence length')
    parser.add_argument('--classification_model', type=str, default="FCN", choices=["FCN", "TF"], help='classification model')

    parser.add_argument('--num_feat', type=int, default=4, help='number of features in dataset')
    parser.add_argument('--num_class', type=int, default=2, help='number of classification classes')

    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')

    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument("--log", "-l", action="store_true", help='use this flag to save log files')

    return parser.parse_args()

if __name__ =="__main__":

    args = get_argumets()

    logger = logging.getLogger("Fore_Auth {} U{}_sl{}_ll{}_pl{}".format(
        args.classification_model,
        str(args.user_id).zfill(2), 
        str(args.seq_len).zfill(2), 
        str(args.label_len).zfill(2), 
        str(args.pred_len).zfill(2)
        ))
    logger.setLevel(logging.DEBUG)
    fmt = "[%(name)s] %(levelname)s>>> %(message)s"
    formatter = logging.Formatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if args.log:
        log_out_to = os.path.join(args.out_root, "log", str(args.user_id).zfill(2))
        if not os.path.exists(log_out_to):
            os.makedirs(log_out_to)

        fh = logging.FileHandler(os.path.join(
            log_out_to,
            "train_userID_{}_sl{}_ll{}_pl{}_{}.log".format(
                str(args.user_id).zfill(2), 
                str(args.seq_len).zfill(2),
                str(args.label_len).zfill(2),
                str(args.pred_len).zfill(2),
                args.classification_model
                )
            ), 
            'w'
            )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.debug("Classification Model: {}".format(args.classification_model))
    logger.debug("Sequence length:      {}".format(args.seq_len))
    logger.debug("Label length:         {}".format(args.label_len))
    logger.debug("Prediction length:    {}".format(args.pred_len))
    logger.debug("Number of epoch:      {}".format(args.max_epoch))
    logger.debug("Batch size:           {}".format(args.batch_size))
    logger.debug("Learning rate:        {}".format(args.learning_rate))

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    logger.info("Using device:          {}\n".format(device))

    train_data = Fore_Auth(
        root_path = args.data_root, 
        flag = 'train', 
        user_id = args.user_id,
        stride = 1,
        forecasting_sizes = [args.seq_len, args.label_len, args.pred_len],
    )

    train_loader = DataLoader(
        train_data,
        batch_size = args.batch_size,
        shuffle = True,
        drop_last = False
    )

    test_data = Fore_Auth(
        root_path = args.data_root, 
        flag = 'test', 
        user_id = args.user_id,
        stride = 1,
        forecasting_sizes = [args.seq_len, args.label_len, args.pred_len],
    )
    test_loader = DataLoader(
        test_data,
        batch_size = args.batch_size,
        shuffle = True,
        drop_last = True
    )

    model = Fore_Auth_Model(
        enc_in=args.num_feat, dec_in=args.num_feat, c_out=args.num_feat, 
        seq_len=args.seq_len, label_len=args.label_len, out_len=args.pred_len,
        num_feats=args.num_feat, num_class=args.num_class, classification_model=args.classification_model,
        d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers, d_layers=args.d_layers, 
        d_ff=args.d_ff, dropout=args.dropout, activation='gelu', output_attention = True,
        device=device
    ).to(device)


    model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion_mse =  nn.MSELoss().to(device)
    criterion_bce_pressure =  nn.BCELoss().to(device)
    criterion_bce_classification =  nn.BCELoss().to(device)

    checkpoint_path = os.path.join(args.out_root, 'checkpoints', str(args.user_id).zfill(2))
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    best_epoch_acc = 0
    global_test_acc = -np.inf
    show_digit = 5
    for epoch in tqdm.tqdm(range(args.max_epoch)):
        model.train()
        epoch_loss = 0
        for i, (en_in, en_time, de_in, de_time, _, label) in enumerate(train_loader):
            label = torch.eye(2)[label.long(), :].to(device)

            model_optim.zero_grad()

            en_in = en_in.float().to(device)
            en_time = en_time.float().to(device)

            true = de_in[:,args.label_len:,:].float().to(device) # true value for forecasting

            # zero padding for decoder input (for forecasting part)
            dec_inp = torch.zeros([de_in.shape[0], args.pred_len, de_in.shape[-1]]).float()
            de_in = torch.cat([de_in[:,:args.label_len,:], dec_inp], dim=1).float().to(device)
            de_time = de_time.float().to(device)

            if args.classification_model == "FCN":
                forecast_attn_map, forecasting_output, classification_output = model(en_in, en_time, de_in, de_time)
            elif args.classification_model == "TF":
                forecast_attn_map, forecasting_output, classification_output = model(en_in, en_time, de_in, de_time)
                classification_output = classification_output[0]
                attn_maps = classification_output[1]
            else:
                raise ValueError("classification model not supported")

            mse_loss = criterion_mse(forecasting_output[:, :, :-1], true[:, :, :-1])    # trajectory loss

            m = nn.Sigmoid()
            pred_pressure = torch.flatten(forecasting_output[:, :, -1])
            pred_pressure = m(pred_pressure)
            true_pressure = torch.flatten(true[:, :, -1])
            condition = true_pressure <= 0.5
            true_pressure = torch.where(condition, torch.tensor(0., dtype=torch.float32).to(device), torch.tensor(1., dtype=torch.float32).to(device))

            bce_pressure_loss = criterion_bce_pressure(pred_pressure, true_pressure)

            
            mm = nn.Sigmoid()
            bce_class_loss = criterion_bce_classification(mm(classification_output), label)

            loss = 0.8 * mse_loss + 0.1 * bce_pressure_loss + bce_class_loss

            loss.backward()
            model_optim.step()
            epoch_loss += loss.item()

        logger.info("Epoch {}/{} traj loss: {} | pressure loss: {} | classification loss: {} | total loss: {}".format(
                epoch + 1, args.max_epoch,  round(mse_loss.item(), show_digit), 
                round(bce_pressure_loss.item(), show_digit),  round(bce_class_loss.item(), show_digit),  
                round(epoch_loss / len(train_loader), show_digit)
                )
            )

        # testing...
        correct_t = 0
        total_t = 0
        with torch.no_grad():
            model.eval()
            all_labels_t, all_preds_t = [], []
            
            for i, (en_in_t, en_time_t, de_in_t, de_time_t, _, label_t) in enumerate(test_loader):    
                all_labels_t.append(label_t.tolist())     

                label_t = label_t.to(device)    

                en_in_t = en_in_t.float().to(device)
                en_time_t = en_time_t.float().to(device)

                true_t = de_in_t[:,args.label_len:,:].float().to(device) # true value for forecasting

                # zero padding for decoder input (for forecasting part)
                dec_inp_t = torch.zeros([de_in_t.shape[0], args.pred_len, de_in.shape[-1]]).float()

                de_in_t = torch.cat([de_in_t[:,:args.label_len,:], dec_inp_t], dim=1).float().to(device)
                de_time_t = de_time_t.float().to(device)

                if args.classification_model == "FCN":
                    forecast_attn_map, forecasting_output, classification_output = model(en_in_t, en_time_t, de_in_t, de_time_t)
                elif args.classification_model == "TF":
                    forecast_attn_map, forecasting_output, classification_output = model(en_in_t, en_time_t, de_in_t, de_time_t)
                    classification_output = classification_output[0]
                    attn_maps = classification_output[1]
                else:
                    raise ValueError("classification model not supported")

                _, pred_t = torch.max(classification_output, dim=1)
                all_preds_t.append(pred_t.cpu().tolist())
                correct_t += torch.sum(pred_t==label_t).item()
                total_t += label_t.size(0)

            flatten_list = lambda y:[x for a in y for x in flatten_list(a)] if type(y) is list else [y]
            all_labels_t = flatten_list(all_labels_t)
            all_preds_t = flatten_list(all_preds_t)
            eer = compute_eer(all_labels_t, all_preds_t)

            if correct_t > global_test_acc:
                global_test_acc = correct_t
                best_epoch_acc = epoch + 1
                torch.save(
                    model.state_dict(), 
                    os.path.join(
                        checkpoint_path,
                        "Fore_Auth_{}_userID_{}_sl{}_ll{}_pl{}.pth".format(
                            args.classification_model, 
                            str(args.user_id).zfill(2),
                            str(args.seq_len).zfill(2), 
                            str(args.label_len).zfill(2), 
                            str(args.pred_len).zfill(2)
                        ),
                    )
                )
            logger.debug("Testing in epoch {}/{} with testing ACC {}.".format(epoch+1, args.max_epoch, 100 * correct_t / total_t))
            logger.debug("Testing in epoch {}/{} with testing EER {}.".format(epoch+1, args.max_epoch, 100 * eer))


    logger.info("Training finished.")
    logger.info("Best testing ACC in epoch {} with testing ACC {}.".format(best_epoch_acc, 100 * global_test_acc / total_t))
    torch.cuda.empty_cache()
