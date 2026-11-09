import os
from data_set import MyDataset
from torch.utils.data import DataLoader
import torch
import logging
from tqdm import tqdm, trange
from sklearn import metrics
import wandb

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(args, model, device, train_data, dev_data, test_data, processor):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    train_loader = DataLoader(dataset=train_data,
                              batch_size=args.train_batch_size,
                              collate_fn=MyDataset.collate_func,
                              shuffle=True)
    total_steps = int(len(train_loader) * args.num_train_epochs)
    model.to(device)

    if args.optimizer_name == 'adafactor':
        from transformers.optimization import Adafactor, AdafactorSchedule

        print('Use Adafactor Optimizer for Training.')
        optimizer = Adafactor(
            model.parameters(),
            # lr=1e-3,
            # eps=(1e-30, 1e-3),
            # clip_threshold=1.0,
            # decay_rate=-0.8,
            # beta1=None,
            lr=None,
            weight_decay=args.weight_decay,
            relative_step=True,
            scale_parameter=True,
            warmup_init=True
        )
        scheduler = AdafactorSchedule(optimizer)
    elif args.optimizer_name == 'adam':
        print('Use AdamW Optimizer for Training.')
        from transformers.optimization import AdamW, get_linear_schedule_with_warmup
        if args.model == 'MV_CLIP':
            clip_params = list(map(id, model.model.parameters()))
            base_params = filter(lambda p: id(p) not in clip_params, model.parameters())
            optimizer = AdamW([
                    {"params": base_params},
                    {"params": model.model.parameters(),"lr": args.clip_learning_rate}
                    ], lr=args.learning_rate, weight_decay=args.weight_decay)

            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                    num_training_steps=total_steps)
        else:
            optimizer = optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    else:
        raise Exception('Wrong Optimizer Name!!!')


    max_acc = float('-inf')
    best_epoch = None
    path_to_save = os.path.join(args.output_dir, args.model)
    os.makedirs(path_to_save, exist_ok=True)
    checkpoint_path = os.path.join(path_to_save, 'model.pt')

    for i_epoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        sum_loss = 0.
        sum_step = 0
        train_targets = []
        train_predictions = []

        iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)", disable=False)
        model.train()

        for step, batch in enumerate(iter_bar):
            text_list, image_list, label_list, id_list = batch
            if args.model == 'MV_CLIP':
                inputs = processor(text=text_list, images=image_list, padding='max_length', truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
                labels = torch.tensor(label_list).to(device)

            loss, score = model(inputs,labels=labels)
            sum_loss += loss.item()
            sum_step += 1
            train_targets.extend(labels.detach().cpu().tolist())
            train_predictions.extend(torch.argmax(score.detach(), -1).cpu().tolist())

            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
            loss.backward()
            optimizer.step()
            if args.optimizer_name == 'adam':
                scheduler.step() 
            optimizer.zero_grad()

        epoch = i_epoch + 1
        train_metrics = calculate_metrics(
            train_targets, train_predictions, sum_loss / sum_step
        )
        dev_metrics = evaluate(args, model, device, dev_data, processor)
        wandb.log({
            'epoch': epoch,
            **{f'train/{name}': value for name, value in train_metrics.items()},
            **{f'dev/{name}': value for name, value in dev_metrics.items()},
        })
        logger.info(
            "epoch=%d train_loss=%.6f train_accuracy=%.6f train_macro_f1=%.6f "
            "dev_loss=%.6f dev_accuracy=%.6f dev_macro_f1=%.6f",
            epoch,
            train_metrics['loss'],
            train_metrics['accuracy'],
            train_metrics['macro_f1'],
            dev_metrics['loss'],
            dev_metrics['accuracy'],
            dev_metrics['macro_f1'],
        )

        if dev_metrics['accuracy'] > max_acc:
            max_acc = dev_metrics['accuracy']
            best_epoch = epoch

            model_to_save = (model.module if hasattr(model, "module") else model)
            torch.save(model_to_save.state_dict(), checkpoint_path)

            artifact = wandb.Artifact(
                name=f'{args.model}-checkpoint',
                type='model',
                description='Checkpoint selected by development-set accuracy',
                metadata={
                    'epoch': epoch,
                    **{f'dev_{name}': value for name, value in dev_metrics.items()},
                },
            )
            artifact.add_file(checkpoint_path, name='model.pt')
            wandb.log_artifact(artifact)
            logger.info("Saved and logged improved checkpoint from epoch %d", epoch)

        torch.cuda.empty_cache()

    if best_epoch is None:
        raise RuntimeError("Training finished without producing a checkpoint")

    model_to_load = (model.module if hasattr(model, "module") else model)
    model_to_load.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics = evaluate(args, model, device, test_data, processor)
    wandb.log({
        'best_epoch': best_epoch,
        **{f'test/{name}': value for name, value in test_metrics.items()},
    })
    wandb.run.summary['best_epoch'] = best_epoch
    wandb.run.summary['best_dev_accuracy'] = max_acc
    logger.info(
        "best_epoch=%d test_loss=%.6f test_accuracy=%.6f test_macro_f1=%.6f",
        best_epoch,
        test_metrics['loss'],
        test_metrics['accuracy'],
        test_metrics['macro_f1'],
    )
    logger.info('Train done')


def calculate_metrics(targets, predictions, loss):
    return {
        'loss': loss,
        'accuracy': metrics.accuracy_score(targets, predictions),
        'f1': metrics.f1_score(targets, predictions, zero_division=0),
        'precision': metrics.precision_score(targets, predictions, zero_division=0),
        'recall': metrics.recall_score(targets, predictions, zero_division=0),
        'macro_f1': metrics.f1_score(
            targets, predictions, labels=[0, 1], average='macro', zero_division=0
        ),
        'macro_precision': metrics.precision_score(
            targets, predictions, labels=[0, 1], average='macro', zero_division=0
        ),
        'macro_recall': metrics.recall_score(
            targets, predictions, labels=[0, 1], average='macro', zero_division=0
        ),
    }


def evaluate(args, model, device, data, processor):
    data_loader = DataLoader(
        data,
        batch_size=args.dev_batch_size,
        collate_fn=MyDataset.collate_func,
        shuffle=False,
    )
    targets = []
    predictions = []
    sum_loss = 0.
    sum_step = 0

    model.eval()
    with torch.no_grad():
        for text_list, image_list, label_list, id_list in data_loader:
            if args.model == 'MV_CLIP':
                inputs = processor(
                    text=text_list,
                    images=image_list,
                    padding='max_length',
                    truncation=True,
                    max_length=args.max_len,
                    return_tensors="pt",
                ).to(device)
                labels = torch.tensor(label_list).to(device)

            loss, outputs = model(inputs, labels=labels)
            sum_loss += loss.item()
            sum_step += 1
            targets.extend(labels.cpu().tolist())
            predictions.extend(torch.argmax(outputs, -1).cpu().tolist())

    return calculate_metrics(targets, predictions, sum_loss / sum_step)
