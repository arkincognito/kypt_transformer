import os

import glob
import configargparse
from common.utils.vis import *
from test import main


def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--gpu_ids", type=str, dest="gpu_ids", default="0")
    parser.add_argument(
        "--annot_subset", type=str, dest="annot_subset", default="all", help="all/human_annot/machine_annot"
    )
    parser.add_argument("--test_set", type=str, dest="test_set", default="test", help="Split type (test/train/val)")
    parser.add_argument("--use_big_decoder", action="store_true", help="Use Big Decoder for U-Net")
    parser.add_argument("--dec_layers", type=int, default=6, help="Number of Cross-attention layers")
    args = parser.parse_args()
    args.capture, args.camera, args.seq_name = None, None, None

    cfg.use_big_decoder = args.use_big_decoder
    cfg.dec_layers = args.dec_layers
    cfg.calc_mutliscale_dim(cfg.use_big_decoder, cfg.resnet_type)

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if "-" in args.gpu_ids:
        gpus = args.gpu_ids.split("-")
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ",".join(map(lambda x: str(x), list(range(*gpus))))

    # assert args.test_epoch, 'Test epoch is required.'
    return args


def run_batch_test():
    args = parse_args()
    cfg.set_args(args.gpu_ids, "")
    for path in glob.glob(
        "/home/plask/wonho/hand_pose_estimation/kypt_transformer/output/model_dump/base/snapshot_*.pth.tar"
    ):
        print(f"path:{path}")
        main(path)


if __name__ == "__main__":
    run_batch_test()
