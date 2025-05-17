import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='root dir for data')
    parser.add_argument('--Dataset_name', type=str, help='experiment_name')
    parser.add_argument('--num_classes', type=int,
                        default=2, help='output channel of network')
    parser.add_argument('--max_epochs', type=int,
                        default=2000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=24, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.1,
                        help='segmentation network learning rate')
    parser.add_argument('--img_size', type=list,
                        default=[224, 224], help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')

    return parser


