import argparse
import torch
from model import build_model

def get_args_parser():
    parser = argparse.ArgumentParser('Set transunet', add_help=False)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=100, type=float)
    parser.add_argument('--batch_size', default=24, type=int)    
    parser.add_argument('--epochs', default=100, type=int)
    
    # Backbone
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--layers', default=True, type=bool)
    
    # Embedding
    parser.add_argument('--patch_size', default=1, type=int)
    parser.add_argument('--num_patches', default=16, type=int)
    parser.add_argument('--d_model', default=2048, type=int)
    parser.add_argument('--input_dim', default=784, type=int)
    parser.add_argument('--hidden_dim', default=2048, type=int)
    
    # Transformer
    parser.add_argument('--n_head', default=4, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--num_layers', default=12, type=int)
    
    parser.add_argument('--n_classes', default=8, type=int)
    
    
    parser.add_argument('--dataset_file', default='NG', type=str)
    parser.add_argument('--NG_path', type=str)
    parser.add_argument('--output_dir',default='', type=str)
    parser.add_argument('--seed', default=777, type=int)
    
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    
    return parser
    
def main(args):


    model= build_model(args)
    x = torch.randn(1,3,256,256) 
    print(model(x).shape)   
if __name__ == '__main__':
    parser = argparse.ArgumentParser('TransUnet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    main(args)

