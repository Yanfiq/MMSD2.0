import os
import subprocess
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from model import MV_CLIP
from train import train
from data_set import MyDataset
import torch
import argparse
import random
import numpy as np
from transformers import CLIPProcessor
import wandb
import pickle
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, help='device number')
    parser.add_argument('--model', default='MV_CLIP', type=str, help='the model name', choices=['MV_CLIP'])
    parser.add_argument('--text_name', default='text_json_final', type=str, help='the text data folder name')
    parser.add_argument('--simple_linear', default=False, type=bool, help='linear implementation choice')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='number of train epoched')
    parser.add_argument('--train_batch_size', default=32, type=int, help='batch size in train phase')
    parser.add_argument('--dev_batch_size', default=32, type=int, help='batch size in dev phase')
    parser.add_argument('--label_number', default=2, type=int, help='the number of classification labels')
    parser.add_argument('--text_size', default=512, type=int, help='text hidden size')
    parser.add_argument('--image_size', default=768, type=int, help='image hidden size')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--optimizer_name", type=str, default='adam',
                        help="use which optimizer to train the model.")
    parser.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate for modules expect CLIP')
    parser.add_argument('--clip_learning_rate', default=1e-6, type=float, help='learning rate for CLIP')
    parser.add_argument('--max_len', default=77, type=int, help='max len of text based on CLIP')
    parser.add_argument('--layers', default=3, type=int, help='number of transform layers')
    parser.add_argument('--max_grad_norm', default=5.0, type=float, help='grad clip norm')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay')
    parser.add_argument('--warmup_proportion', default=0.2, type=float, help='warm up proportion')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--output_dir', default='../output_dir/', type=str, help='the output path')
    parser.add_argument('--limit', default=None, type=int, help='the limited number of training examples')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--wait_for_gpu', action='store_true',
                        help='wait until the selected GPU has enough free VRAM and is idle')
    parser.add_argument('--min_free_vram_mb', type=int, default=12000,
                        help='minimum free GPU memory required when --wait_for_gpu is used')
    parser.add_argument('--max_gpu_utilization', type=int, default=10,
                        help='maximum GPU utilization percentage considered idle')
    parser.add_argument('--gpu_poll_interval', type=float, default=30.0,
                        help='seconds between GPU availability checks')
    parser.add_argument('--gpu_wait_timeout', type=float, default=0.0,
                        help='maximum seconds to wait; 0 waits indefinitely')
    return parser.parse_args()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def wait_for_gpu(device, min_free_vram_mb, max_utilization, poll_interval, timeout):
    """Wait for two consecutive acceptable nvidia-smi readings."""
    if int(device) < 0:
        print("CPU mode selected; skipping the GPU availability check.", flush=True)
        return
    if min_free_vram_mb < 0:
        raise ValueError("--min_free_vram_mb must be at least 0")
    if not 0 <= max_utilization <= 100:
        raise ValueError("--max_gpu_utilization must be between 0 and 100")
    if poll_interval <= 0:
        raise ValueError("--gpu_poll_interval must be greater than 0")
    if timeout < 0:
        raise ValueError("--gpu_wait_timeout must be at least 0")

    command = [
        "nvidia-smi",
        f"--id={device}",
        "--query-gpu=memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    started_at = time.monotonic()
    acceptable_readings = 0

    while True:
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            free_vram_mb, utilization = (
                int(value.strip()) for value in result.stdout.strip().split(",")
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "--wait_for_gpu requires nvidia-smi, but it was not found"
            ) from exc
        except (subprocess.CalledProcessError, ValueError) as exc:
            details = getattr(exc, "stderr", "") or str(exc)
            raise RuntimeError(f"Could not query GPU {device}: {details.strip()}") from exc

        available = (
            free_vram_mb >= min_free_vram_mb
            and utilization <= max_utilization
        )
        acceptable_readings = acceptable_readings + 1 if available else 0
        elapsed = time.monotonic() - started_at
        print(
            f"GPU {device}: {free_vram_mb} MiB free, {utilization}% utilized "
            f"(need >= {min_free_vram_mb} MiB and <= {max_utilization}%)",
            flush=True,
        )

        # Two readings prevent starting during a brief dip between other workloads.
        if acceptable_readings >= 2:
            print(f"GPU {device} is available; starting training.", flush=True)
            return
        if timeout and elapsed >= timeout:
            raise TimeoutError(
                f"GPU {device} did not become available within {timeout:g} seconds"
            )
        time.sleep(poll_interval)


def main():
    args = set_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.wait_for_gpu:
        wait_for_gpu(
            args.device,
            args.min_free_vram_mb,
            args.max_gpu_utilization,
            args.gpu_poll_interval,
            args.gpu_wait_timeout,
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")

    seed_everything(args.seed)

    wandb.init(
        project="MMSD2.0",
        notes="mm",
        tags=["mm"],
        config=vars(args),
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("dev/*", step_metric="epoch")
    wandb.watch_called = False  

    train_data = MyDataset(mode='train', text_name=args.text_name, limit=None)
    dev_data = MyDataset(mode='valid', text_name=args.text_name, limit=None)
    test_data = MyDataset(mode='test', text_name=args.text_name, limit=None)

    if args.model == 'MV_CLIP':
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = MV_CLIP(args)
    else:
        raise RuntimeError('Error model name!')

    model.to(device)
    wandb.watch(model, log="all")

    train(args, model, device, train_data, dev_data, test_data, processor)
    wandb.finish()



if __name__ == '__main__':
    main()
