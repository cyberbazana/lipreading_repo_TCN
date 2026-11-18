import os
import glob
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from lipreading.utils import load_json, save2npz, load_model
from lipreading.model import Lipreading
from lipreading.dataloaders import get_preprocessing_pipelines


def load_args():
    parser = argparse.ArgumentParser(description='Batch lipreading embedding extraction')
    parser.add_argument('--modality', default='video', choices=['video', 'audio'])
    parser.add_argument('--config-path', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--mouth-patch-dir', type=str, required=True,
                        help='Folder with .npz files (each npz must contain key "data", shape TxHxW)')
    parser.add_argument('--mouth-embedding-out-dir', type=str, required=True)
    parser.add_argument('--recursive', action='store_true',
                        help='Search .npz files recursively (keeps relative subfolder structure)')
    parser.add_argument('--label-path', type=str, default='./labels/500WordsSortedList.txt')
    parser.add_argument('--allow-size-mismatch', action='store_true')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    return parser.parse_args()


def get_model_from_json(args, device):
    args_loaded = load_json(args.config_path)

    if args_loaded.get('tcn_num_layers', ''):
        tcn_options = {'num_layers': args_loaded['tcn_num_layers'],
                       'kernel_size': args_loaded['tcn_kernel_size'],
                       'dropout': args_loaded['tcn_dropout'],
                       'dwpw': args_loaded['tcn_dwpw'],
                       'width_mult': args_loaded['tcn_width_mult']}
    else:
        tcn_options = {}

    if args_loaded.get('densetcn_block_config', ''):
        densetcn_options = {'block_config': args_loaded['densetcn_block_config'],
                            'growth_rate_set': args_loaded['densetcn_growth_rate_set'],
                            'reduced_size': args_loaded['densetcn_reduced_size'],
                            'kernel_size_set': args_loaded['densetcn_kernel_size_set'],
                            'dilation_size_set': args_loaded['densetcn_dilation_size_set'],
                            'squeeze_excitation': args_loaded['densetcn_se'],
                            'dropout': args_loaded['densetcn_dropout']}
    else:
        densetcn_options = {}

    model = Lipreading(
        modality=args.modality,
        num_classes=500,
        tcn_options=tcn_options,
        densetcn_options=densetcn_options,
        backbone_type=args_loaded['backbone_type'],
        relu_type=args_loaded['relu_type'],
        width_mult=args_loaded['width_mult'],
        use_boundary=args_loaded.get('use_boundary', False),
        extract_feats=True
    ).to(device)
    return model


class NpzDataset(Dataset):
    def __init__(self, files, root_dir, preprocessing_func):
        self.files = files
        self.root_dir = root_dir
        self.preprocessing_func = preprocessing_func

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        data = np.load(fp)['data']
        data = self.preprocessing_func(data)           
        data = np.ascontiguousarray(data, dtype=np.float32)
        rel = os.path.relpath(fp, self.root_dir)
        return data, rel


def collate_fn(batch):
    datas, rels = zip(*batch)
    lengths = [d.shape[0] for d in datas]
    t_max = max(lengths)
    h, w = datas[0].shape[1], datas[0].shape[2]
    out = np.zeros((len(datas), 1, t_max, h, w), dtype=np.float32)
    for i, d in enumerate(datas):
        out[i, 0, :d.shape[0]] = d
    return torch.from_numpy(out), lengths, list(rels)


def main():
    args = load_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    model = get_model_from_json(args, device)
    model = load_model(args.model_path, model, allow_size_mismatch=args.allow_size_mismatch)
    model.eval()
    preprocessing_func = get_preprocessing_pipelines(args.modality)['test']
    if args.recursive:
        files = glob.glob(os.path.join(args.mouth_patch_dir, '**', '*.npz'), recursive=True)
    else:
        files = glob.glob(os.path.join(args.mouth_patch_dir, '*.npz'))
    files.sort()
    print(f'Found {len(files)} .npz files')
    os.makedirs(args.mouth_embedding_out_dir, exist_ok=True)
    dataset = NpzDataset(files, args.mouth_patch_dir, preprocessing_func)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
    )

    n_done = 0
    with torch.no_grad():
        for inp, lengths, rels in loader:
            inp = inp.to(device, non_blocking=True)
            feats = model(inp, lengths=lengths)          
            feats = feats.cpu().numpy()
            for i, rel in enumerate(rels):
                out_fp = os.path.join(args.mouth_embedding_out_dir, os.path.splitext(rel)[0] + '.npz')
                os.makedirs(os.path.dirname(out_fp) or '.', exist_ok=True)
                save2npz(out_fp, data=feats[i])
            n_done += len(rels)
            print(f'\r{n_done}/{len(files)}', end='', flush=True)

    print('\nDone.')


if __name__ == '__main__':
    main()