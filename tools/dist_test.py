import argparse
import copy
import json
import os
import sys

try:
    import apex
except:
    print("No APEX!")
import numpy as np
import torch
import yaml
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import get_dist_info, load_checkpoint, save_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from torch.nn.parallel import DistributedDataParallel
import pickle 
import time
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import seaborn as sns
tsne = TSNE()

def plot_vecs_n_labels(v, labels, fname):
    fig = plt.figure(figsize = (10,10))
    plt.axis('off')
    sns.set_style('darkgrid')
    sns.scatterplot(v[:,0], v[:,1], hue=labels, legend='full', palette=sns.color_palette("bright", 10))
    plt.legend(['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck'])
    plt.savefig(fname)

def save_pred(pred, root):
    with open(os.path.join(root, "prediction.pkl"), "wb") as f:
        pickle.dump(pred, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", required=True, help="the dir to save logs and models")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    parser.add_argument(
        "--txt_result",
        type=bool,
        default=False,
        help="whether to save results to standard KITTI format of txt type",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--testset", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--speed_test", type=str, default=None,
                        help="event or profiler (pytorch version 1.9 is needed for profiler")
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.cpu and not args.speed_test is None:
        torch.set_num_threads(16)
    elif not args.speed_test is None:
        torch.set_num_threads(16)

    return args


def main():

    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()
    else:
        cfg.gpus = args.gpus

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed testing: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    if args.testset:
        print("Use Test Set")
        dataset = build_dataset(cfg.data.test)
    else:
        print("Use Val Set")
        dataset = build_dataset(cfg.data.val)

    data_loader = build_dataloader(
        dataset,
        batch_size=cfg.data.samples_per_gpu if args.speed_test is None else 1,
        workers_per_gpu=cfg.data.workers_per_gpu if args.speed_test is None else 1,
        dist=distributed,
        shuffle=False,
    )
    if not args.speed_test and not args.checkpoint is None:
        checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    # put model on gpus
    if distributed:
        model = apex.parallel.convert_syncbn_model(model)
        model = DistributedDataParallel(
            model.cuda(cfg.local_rank),
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            # broadcast_buffers=False,
            find_unused_parameters=True,
        )
    else:
        # model = fuse_bn_recursively(model)
        if args.cpu:
            model = model.to('cpu')
        else:
            model = model.cuda()

    model.eval()
    mode = "val"

    logger.info(f"work dir: {args.work_dir}")
    if cfg.local_rank == 0:
        prog_bar = torchie.ProgressBar(len(data_loader.dataset) // cfg.gpus)

    detections = {}
    cpu_device = torch.device("cpu")

    elapse_time = None
    if args.speed_test == 'event':
        elapse_time = {'elapse_start': torch.cuda.Event(enable_timing=True),
                       'elapse_end': torch.cuda.Event(enable_timing=True),
                       'reader': float(0),
                       'backbone': float(0),
                       'neck': float(0),
                       'head': float(0),
                       'start': int(len(dataset) / 10),
                       'end': int(len(dataset) * 9 / 10)}
    elif args.speed_test == 'profiler':
        import torch.profiler as prof
        acitivities = [
            prof.ProfilerActivity.CPU,
            prof.ProfilerActivity.CUDA,
        ]
        schedule = prof.schedule(
            wait=0,
            warmup=int(len(dataset) / 3),
            active=int(len(dataset) * 2 / 3),
        )
        profiler = prof.profile(activities=acitivities,
                                schedule=schedule,
                                on_trace_ready=lambda p:print(p.key_averages().table(
                                    sort_by="self_cuda_time_total", row_limit=-1
                                )))

    count = 0
    # alpha_cumsum = torch.zeros([11])
    # beta_cumsum = torch.zeros([11])
    # cumsum = torch.zeros([11])
    avgval = np.zeros(4)
    count = 0
    for i, data_batch in enumerate(data_loader):
        if args.speed_test == 'event':
            elapse_time['iter'] = i
        with torch.no_grad():
            outputs = batch_processor(
                model, data_batch, train_mode=False,
                local_rank=args.local_rank if not args.cpu else 'cpu',
                elapse_time=elapse_time
            )
            # _, elapse_time = outputs
            if args.speed_test == None:
                if type(outputs) is tuple:
                    outputs, sim = outputs
                    for i, val in enumerate(sim):
                        avgval[i] += val
                    count += 1
                for output in outputs:
                    token = output["metadata"]["token"]
                    for k, v in output.items():
                        if k not in [
                            "metadata",
                        ]:
                            output[k] = v.to(cpu_device)
                    detections.update(
                        {token: output,}
                    )
            if args.local_rank == 0:
                prog_bar.update()
        if args.speed_test == 'profilers':
            profiler.step()
            if args.local_rank == 0:
                prog_bar.update()

    synchronize()

    if args.speed_test == 'event':
        total_time = (elapse_time['reader'] + elapse_time['backbone'] + elapse_time['neck'] + elapse_time['head']) \
                     / ((elapse_time['end'] - elapse_time['start']) * 1000)

        print('\n=========elapse time===========')
        print('Time per image: {:.6f} sec\n'
              'Reader time: {:.6f} sec\n'
              'Backbone time: {:.6f} sec\n'
              'Neck time: {:.6f} sec\n'
              'Head time: {:.6f} sec\n'
              .format(total_time,
                      elapse_time['reader'] / ((elapse_time['end'] - elapse_time['start']) * 1000),
                      elapse_time['backbone'] / ((elapse_time['end'] - elapse_time['start']) * 1000),
                      elapse_time['neck'] / ((elapse_time['end'] - elapse_time['start']) * 1000),
                      elapse_time['head'] / ((elapse_time['end'] - elapse_time['start']) * 1000)))
        print('===============================')
    elif args.speed_test is None:
        all_predictions = all_gather(detections)

        if args.local_rank != 0:
            return

        predictions = {}
        for p in all_predictions:
            predictions.update(p)

        if not os.path.exists(args.work_dir):
            os.makedirs(args.work_dir)

        save_pred(predictions, args.work_dir)

        result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), output_dir=args.work_dir, testset=args.testset)

        if result_dict is not None:
            for k, v in result_dict["results"].items():
                print(f"Evaluation {k}: {v}")

        if args.txt_result:
            assert False, "No longer support kitti"
    elif args.speed_test == 'profiler':
        print(profiler.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1
        ))
    # print(alpha_cumsum/len(data_loader))
    # print(beta_cumsum/len(data_loader))
    # print(cumsum / len(data_loader))

if __name__ == "__main__":
    main()
