'''
Author: Aman
Date: 2022-04-17 22:55:25
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2022-04-17 23:09:08
'''


import argparse
import logging
import os
import random
import time
import pickle

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter  

from configs import data_config, new_layers_config
from model import MyBERT
from MyDataset import MyDataset
from loss import MyLoss
from utils import EarlyStopping, format_time

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


parser = argparse.ArgumentParser()
parser.add_argument("--device_ids", default="0,1,2,3", type=str, help="GPU device ids")
parser.add_argument("--model_name", default="bert-base-chinese", type=str, help="Model name") # bert-base-chinese / hfl/chinese-bert-wwm-ext
parser.add_argument("--batch_size", default=48, type=int, help="Batch size")
parser.add_argument("--eval_batch_size", default=48, type=int, help="Test batch size")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs")
parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--num_workers", default=8, type=int, help="Number of workers")
parser.add_argument("--log_interval", default=100, type=int, help="Log interval")
parser.add_argument("--eval_interval_ratio", default=0.26, type=float, help="Eval once every interval ratio of training data")
parser.add_argument("--data_train_val", default="./datasets/dataset_train_val.pkl", type=str, help="Train val data")
parser.add_argument("--save_model", default=True, type=bool, help="Save model or not")
parser.add_argument("--save_path", default=".", type=str, help="Save directory")
parser.add_argument("--log_path", default=".", type=str, help="Log directory")
parser.add_argument("--tensorboard_log_dir", default=".", type=str, help="Tensorboard log directory")
parser.add_argument("--study_name", default="default", type=str, help="Study name")
parser.add_argument("--storage", default="sqlite:///default.sqlite3", type=str, help="Optuna-dashboard storage path, default=sqlite:///db.sqlite3")
parser.add_argument("--patience", default="3", type=int, help="Early stop patience.")
parser.add_argument("--gradient_accumulation_steps", default="8", type=int, help="Gradient accumulation_steps .")

global args
args = parser.parse_args()
batch_size = args.batch_size
eval_batch_size = args.eval_batch_size
new_layers_config = new_layers_config()
data_config = data_config()
print(args)
logging.basicConfig(filename=args.log_path,
                    level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)-2s - %(filename)-8s : %(lineno)s line - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.info(args)
if not os.path.exists(args.tensorboard_log_dir):
    os.makedirs(args.tensorboard_log_dir)
writer = SummaryWriter(args.tensorboard_log_dir)


EOS = "[#EOS#]"
tokenizer = BertTokenizer.from_pretrained("vocab/vocab.txt", never_split=[EOS])
# tokenizer.vocab['[#EOS#]'] = tokenizer.vocab.pop('[unused1]')


devices = list(eval(args.device_ids))
multi_gpu = False
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name())
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        multi_gpu = True
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy`` and ``torch``.
 
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main(trial=None):
    if trial: # This is for optuna optimization of hyperparameters (learning rate, batch size, etc.)
        args.lr = trial.suggest_discrete_uniform("lr", 2e-5, 1.2e-4, 1e-5)
        # args.lr = trial.suggest_loguniform("lr", 7e-5, 1.2e-4)
        # args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        # args.epochs = trial.suggest_int("epochs", 5, 15)
        # new_layers_config.linear_hidden_size = trial.suggest_categorical("linear_hidden_size", [128, 256, 512])
        # new_layers_config.linear_dropout = trial.suggest_discrete_uniform("linear_dropout", 0.0, 0.5, 0.05)

    model = MyBERT(new_layers_config, args.model_name)

    # unfreeze_layers = ['layer.10','layer.11','bert.pooler','out.']
    # for name, param in model.bert.named_parameters():
    #     param.requires_grad = False
    #     for ele in unfreeze_layers:
    #         if ele in name:
    #             param.requires_grad = True
    #             break

    # freeze all
    # for name, param in model.bert.named_parameters():
    #     param.requires_grad = False

    
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print('* number of parameters: %d' % n_params)
    logger.info('* number of parameters: %d' % n_params)  # compute the number of parameters

    print("Loading data...")
    train_data = MyDataset(args.train_data, tokenizer, data_config)
    valid_data = MyDataset(args.val_data, tokenizer, data_config)
    print("Data loaded.")

    # model.resize_token_embeddings(len(tokenizer))
    # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.

    if multi_gpu:
        model = nn.DataParallel(model, device_ids = devices)
        # model.to(f'cuda:{model.device_ids[0]}')
        model.to(device)
    else:
        model = model.to(device)

    res = train(model, train_data, valid_data)
    return res


def train(model, train_data, valid_data):
    print("Now lr is ", args.lr)
    logger.info('Now lr is %s, epochs is %s, linear_hidden_size is %s, linear_dropout is %s' \
        % (args.lr, args.epochs, new_layers_config.linear_hidden_size, new_layers_config.linear_dropout))
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataset = DataLoader(valid_data, batch_size=eval_batch_size, shuffle=True, num_workers=args.num_workers)


    optimizer = AdamW(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    one_epoch_steps = len(train_dataset)
    training_steps = int(one_epoch_steps * args.epochs)
    print('Total training steps:', training_steps)
    logger.info('* number of training steps: %d' % training_steps) # number of training steps
    # warmup and decay the learning rate
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = int(one_epoch_steps / args.gradient_accumulation_steps * 0.1), 
                                                num_training_steps = training_steps // args.gradient_accumulation_steps)
    # criterion = nn.BCEWithLogitsLoss()
    criterion_CE = nn.CrossEntropyLoss()
    criterion = MyLoss()

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    best_eval_loss = float("inf")
    global_steps = 0
    for epoch in range(args.epochs):
        t1 = time.time()
        torch.cuda.empty_cache()
        print("\nEpoch ", epoch + 1, "/", args.epochs)
        logger.info("Epoch " + str(epoch + 1) + "/" + str(args.epochs))
        avg_loss = 0.0
        model.train()
        # Setting the tqdm progress bar
        epoch_iterator = tqdm(enumerate(train_dataset),
                              desc="%s: %d/%d Epochs >> Steps" % ("Train", epoch + 1, args.epochs),
                              total=len(train_dataset),
                              bar_format="{l_bar}{r_bar}")
        # target_train_dataset = iter(target_train_dataset)
        for step, batch in epoch_iterator:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model.forward(batch)
            ys = batch["labels"]
            # compute loss
            loss = criterion(preds, ys)
            # compute total loss
            loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip gradient
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            
            for param_group in optimizer.param_groups:
                args.lr = param_group['lr']
            epoch_iterator.set_postfix(lr=args.lr, loss=loss.item())  # show the learning rate and loss on the progress bar 
            global_steps += 1
            writer.add_scalar('loss', loss.item(), global_steps)
            if step > 0 and (step + 1) % int(one_epoch_steps * args.eval_interval_ratio) == 0:
                eval_loss, eval_acc = evaluate(model, valid_dataset, criterion)
                logger.info("Epoch: %d, Step: %d/%d, Eval Loss: %.4f, Eval Accuracy: %.2f%%, %.2f%%" % (epoch + 1, step + 1, one_epoch_steps, eval_loss, eval_acc))
                print(" Epoch: %d, Step: %d/%d, Eval Loss: %.4f, Eval Accuracy: %.2f%%, %.2f%%" % (epoch + 1, step + 1, one_epoch_steps, eval_loss, eval_acc))
                writer.add_scalar('eval/loss', eval_loss, global_steps)
                writer.add_scalar('eval/acc', eval_acc, global_steps)
                # Save model
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    if args.save_model:
                        if not os.path.exists(args.save_path):
                            os.makedirs(args.save_path)
                        state = {'model': model.state_dict(), 'args': args} # 'optimizer': optimizer.state_dict(), 
                        torch.save(state, args.save_path + "/model.pth")
                        logger.info("Epoch: %d, Step: %d, Saving Model to \'%s\'." % (epoch + 1, step, args.save_path))
                        print("Epoch: %d, Step: %d, Saving Model to \'%s\'." % (epoch + 1, step, args.save_path))
                # Early stop
                early_stopping(eval_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping!")
                    exit(0)
                model.train()
            avg_loss += loss.item()
            if step > 0 and (step + 1) % args.log_interval == 0:
                logger.info("Epoch: %d, Step: %d/%d, Average loss: %.6f" % (epoch + 1, step + 1, one_epoch_steps, avg_loss / (step + 1)))
        # End of epoch
        writer.add_scalar('loss', loss.item(), global_steps)
        eval_loss, eval_acc = evaluate(model, valid_dataset, criterion)
        writer.add_scalar('eval/loss', eval_loss, global_steps)
        writer.add_scalar('eval/acc', eval_acc, global_steps)
        logger.info("End eval of epoch %d. Eval Loss: %.4f, Eval Accuracy: %.2f%%, %.2f%%" % (epoch + 1, eval_loss, eval_acc))
        print("End eval of epoch %d. Eval Loss: %.4f, Eval Accuracy: %.2f%%, %.2f%%" % (epoch + 1, eval_loss, eval_acc))
        model.train()
        logger.info("Average loss: %.4f  Elapsed time: %s" % (avg_loss / (one_epoch_steps + 1), format_time(time.time()-t1)))
        print("Average loss: %.4f  Elapsed time: %s" % (avg_loss / (one_epoch_steps + 1), format_time(time.time()-t1)))
        if args.save_model:
            state = {'model': model.state_dict(), 'args': args} # 'optimizer': optimizer.state_dict(), 
            torch.save(state, args.save_path + f"/model_{epoch}.pth")
            logger.info("Epoch: %d end. Saving Model to \'%s\'." % (epoch + 1, args.save_path))
            print("Epoch: %d end. Saving Model to \'%s\'." % (epoch + 1, args.save_path))
        # Early stop
        early_stopping(eval_loss, model)
        if early_stopping.early_stop:
            print("Early stopping!")
            exit(0)
    logger.info("Training finished.")
    print("Training finished.")

    return sum(eval_acc) - eval_loss


def evaluate(model, valid_dataset, criterion):
    model.eval()
    eval_loss = 0.0
    eval_acc = 0.0
    y_preds = []
    y_labels = []
    with torch.no_grad():
        epoch_iterator = tqdm(valid_dataset, ncols=100, leave=False)
        for i, batch in enumerate(epoch_iterator):
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model.forward(batch)
            ys = batch["labels"]
            y_preds.append(torch.argmax(preds, dim=1))
            y_labels.append(ys)
            # compute loss
            loss = criterion(preds, ys)
            # compute total loss
            loss = loss.mean()
            eval_loss += loss.item()
    eval_loss /= len(valid_dataset)
    y_preds = torch.cat(y_preds, dim=0)
    y_labels = torch.cat(y_labels, dim=0)
    eval_acc = (y_preds == y_labels).sum().item() / len(y_labels) * 100

    return eval_loss, eval_acc


def optuna_optimize(trial_times):
    study = optuna.create_study(study_name=args.study_name, direction="maximize", storage=args.storage) # storage='sqlite:///db.sqlite3' for optuna-dash
    study.optimize(main, n_trials=trial_times)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":

    time_begin = time.time()
    set_seed(args.seed)

    main()

    # optuna_optimize(20)

    time_end = time.time()
    print("Finished!\nTotal time: %s" % format_time(time_end - time_begin))
    










