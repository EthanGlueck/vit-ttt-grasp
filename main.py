import datetime
import os
import sys
import argparse
import logging
import torch
import torch.utils.data
import torch.optim as optim
from torchsummary import summary
from traning import train, validate
from utils.data import get_dataset
from models.swin import SwinTransformerSys
from models.ggcnn import GGCNN
from models.grconvnet import GenerativeResnet
from models.ttt_swin import SwinTransformerSysTTT
from models.vit_grasp import ViTGraspModel
from models.vit_ttt_grasp import ViTTTTGraspModel


logging.basicConfig(level=logging.INFO)
def parse_args():
    parser = argparse.ArgumentParser(description='TF-Grasp')

    # Network
    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str,default="jacquard", help='Dataset Name ("cornell" or "jacquard")')
    parser.add_argument('--dataset-path', type=str,default="/home/sam/Desktop/cornell" ,help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=0, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1, help='Use RGB image for training (0/1)')
    parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--model', type=str, default='tfgrasp',
                        choices=['tfgrasp', 'ggcnn', 'grconvnet', 'tttswin', 'vitgrasp', 'vittttgrasp'],
                        help='Which model to train: tfgrasp (Swin TF-Grasp) or ggcnn or grconvnet or swin_ttt (Swin with TTT blocks)')
    
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--vis', type=bool, default=False, help='vis')
    parser.add_argument('--epochs', type=int, default=2000, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=200, help='Batches per Epoch')
    parser.add_argument('--val-batches', type=int, default=32, help='Validation Batches')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input image size in pixels (only affects ViT-based models, must be divisible by 16)')
    # Logging etc.
    parser.add_argument('--description', type=str, default='', help='Training description')
    parser.add_argument('--outdir', type=str, default='output/models/', help='Training Output Directory')

    args = parser.parse_args()
    return args
def run():
    args = parse_args()
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))

    save_folder = os.path.join(args.outdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # tb = tensorboardX.SummaryWriter(os.path.join(args.logdir, net_desc))

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)

    # Only ViT-based models support variable image sizes; others stay at 224.
    output_size = args.image_size if args.model in ('vitgrasp', 'vittttgrasp') else 224

    train_dataset = Dataset(args.dataset_path, start=0.0, end=args.split, ds_rotate=args.ds_rotate,
                            random_rotate=True, random_zoom=False,
                            include_depth=args.use_depth, include_rgb=args.use_rgb,
                            output_size=output_size)
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
                          random_rotate=True, random_zoom=False,
                          include_depth=args.use_depth, include_rgb=args.use_rgb,
                          output_size=output_size)
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    logging.info('Done')

    logging.info('Loading Network...')
    input_channels = 1*args.use_depth + 3*args.use_rgb

    #Determines which model to use
    if args.model == 'tfgrasp':
        net = SwinTransformerSys(in_chans=input_channels, embed_dim=48, num_heads=[1, 2, 4, 8])
    elif args.model == 'ggcnn':
        net = GGCNN(input_channels=input_channels)
    elif args.model == 'grconvnet':
        net = GenerativeResnet(input_channels=input_channels, prob=0.0)
    elif args.model == 'tttswin':
        net = SwinTransformerSysTTT(in_chans=input_channels, embed_dim=48, num_heads=[1, 2, 4, 8], 
                                use_conv_branch=True)
    elif args.model == 'vitgrasp':
        net = ViTGraspModel(input_channels=input_channels, pretrained=True, image_size=output_size)
    elif args.model == 'vittttgrasp':
        net = ViTTTTGraspModel(input_channels=input_channels, pretrained=True, use_conv_branch=True, image_size=output_size)

    device = torch.device("cuda:0")
    net = net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=1e-4)
    listy = [x * 2 for x in range(1,1000,5)]
    schedule=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=listy,gamma=0.5)
    logging.info('Done')
    summary(net, (input_channels, output_size, output_size))
    f = open(os.path.join(save_folder, 'net.txt'), 'w')
    sys.stdout = f
    summary(net, (input_channels, output_size, output_size))
    sys.stdout = sys.__stdout__
    f.close()
    best_iou = 0.0
    for epoch in range(args.epochs):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        print("current lr:",optimizer.state_dict()['param_groups'][0]['lr'])
        # for i in range(5000):
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch, vis=args.vis)

        # schedule.step()
        # Run Validation
        logging.info('Validating...')
        test_results = validate(net, device, val_data, args.val_batches)
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct']/(test_results['correct']+test_results['failed'])))
        logging.info('Average Inference Time: {:.4f}s'.format(test_results['avg_inference_time']))

        # Calculate and log FLOPs if model has flops() method
        try:
            if hasattr(net, 'flops'):
                total_flops = net.flops()
                logging.info('FLOPs: {:.2e}'.format(total_flops))
        except Exception as e:
            logging.debug('Could not calculate FLOPs: {}'.format(str(e)))

        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        if epoch%1==0 or iou>best_iou:
            torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
            torch.save(net.state_dict(), os.path.join(save_folder, 'epoch_%02d_iou_%0.2f_statedict.pt' % (epoch, iou)))
        best_iou = iou


if __name__ == '__main__':
    run()
