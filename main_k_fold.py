import datetime
import os
import sys
import argparse
import logging
import cv2
import torch
import torch.utils.data
import torch.optim as optim
from torchsummary import summary
from sklearn.model_selection import KFold
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
    parser.add_argument('--dataset', type=str,default="cornell", help='Dataset Name ("cornell" or "jaquard or multi")')
    parser.add_argument('--dataset-path', type=str,default="/home/sam/Desktop/archive111" ,help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for training (0/1)')
    parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--model', type=str, default='tfgrasp',
                        choices=['tfgrasp', 'ggcnn', 'grconvnet', 'tttswin', 'vitgrasp', 'vittttgrasp'],
                        help='Which model to train: tfgrasp (Swin TF-Grasp) or ggcnn or grconvnet or swin_ttt (Swin with TTT blocks)')

    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=2000, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=500, help='Batches per Epoch')
    parser.add_argument('--val-batches', type=int, default=50, help='Validation Batches')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input image size in pixels (only affects ViT-based models, must be divisible by 16)')
    # Logging etc.
    parser.add_argument('--description', type=str, default='', help='Training description')
    parser.add_argument('--outdir', type=str, default='output/models/', help='Training Output Directory')

    args = parser.parse_args()
    return args


def make_model(args, input_channels, output_size, device):
    """Instantiate a fresh model, optimizer, and scheduler. Called once per fold."""
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
        net = ViTGraspModel(input_channels=input_channels, pretrained=False, image_size=output_size)
    elif args.model == 'vittttgrasp':
        net = ViTTTTGraspModel(input_channels=input_channels, pretrained=False, use_conv_branch=True,
                               image_size=output_size)
    net = net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=5e-4)
    listy = [x * 7 for x in range(1, 1000, 3)]
    schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=listy, gamma=0.6)
    return net, optimizer, schedule


def run():
    args = parse_args()
    # Set-up output directories
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))

    save_folder = os.path.join(args.outdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)

    # Only ViT-based models support variable image sizes; others stay at 224.
    output_size = args.image_size if args.model in ('vitgrasp', 'vittttgrasp') else 224

    dataset = Dataset(args.dataset_path, start=0.0, end=1.0, ds_rotate=args.ds_rotate,
                      random_rotate=True, random_zoom=True,
                      include_depth=args.use_depth, include_rgb=args.use_rgb,
                      output_size=output_size)

    logging.info('Dataset size: {}'.format(len(dataset)))

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)
    logging.info('Done')

    input_channels = 1 * args.use_depth + 3 * args.use_rgb
    device = torch.device("cuda:0")

    # Print model architecture once using a throwaway instance
    logging.info('Loading Network...')
    _net_for_summary, _, _ = make_model(args, input_channels, output_size, device)
    summary(_net_for_summary, (input_channels, output_size, output_size))
    f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    sys.stdout = f
    summary(_net_for_summary, (input_channels, output_size, output_size))
    sys.stdout = sys.__stdout__
    f.close()
    del _net_for_summary
    logging.info('Done')

    # -----------------------------------------------------------------------
    # CORRECTED K-FOLD LOOP
    #
    # The original code (below, commented out) used a single shared model for
    # all folds across all epochs. This causes data leakage: by fold K the
    # model has already been trained on samples that appear in fold K's test
    # set during earlier folds. This produces inflated accuracy (eventually
    # 100% per-fold averages) that does not reflect real generalisation.
    #
    # The corrected version instantiates a fresh model, optimizer, and
    # scheduler for each fold and trains it to convergence independently.
    # The reported accuracy is the mean IW accuracy across the 5 held-out
    # test sets, which is a valid cross-validation estimate.
    # -----------------------------------------------------------------------

    fold_accuracies = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        logging.info('======= FOLD {:d} ======='.format(fold))

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        trainloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, num_workers=args.num_workers,
            sampler=test_subsampler)

        # Fresh model, optimizer, scheduler for this fold
        net, optimizer, schedule = make_model(args, input_channels, output_size, device)

        best_iou_fold = 0.0

        for epoch in range(args.epochs):
            logging.info('Fold {:d} | Epoch {:03d}'.format(fold, epoch))
            print("lr:", optimizer.state_dict()['param_groups'][0]['lr'])

            train(epoch, net, device, trainloader, optimizer, args.batches_per_epoch)
            schedule.step()

            logging.info('Validating...')
            test_results = validate(net, device, testloader, args.val_batches)
            iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
            logging.info('%d/%d = %f' % (test_results['correct'],
                                         test_results['correct'] + test_results['failed'], iou))
            logging.info('Average Inference Time: {:.4f}s'.format(test_results['avg_inference_time']))

            try:
                if hasattr(net, 'flops'):
                    logging.info('FLOPs: {:.2e}'.format(net.flops()))
            except Exception as e:
                logging.debug('Could not calculate FLOPs: {}'.format(str(e)))

            if iou > best_iou_fold or epoch == 0 or (epoch % 50) == 0:
                torch.save(net, os.path.join(
                    save_folder, 'fold_%02d_epoch_%03d_iou_%0.2f' % (fold, epoch, iou)))
                best_iou_fold = iou

        fold_accuracies.append(best_iou_fold)
        logging.info('Fold {:d} best IOU: {:.4f}'.format(fold, best_iou_fold))

    mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    logging.info('======= FINAL MEAN ACCURACY: {:.4f} ======='.format(mean_accuracy))
    print("fold accuracies:", fold_accuracies)
    print("mean accuracy:", mean_accuracy)

    # -----------------------------------------------------------------------
    # ORIGINAL (LEAKY) K-FOLD LOOP — kept for reference, do not use
    # The original implementation trains the same model on all 5 folds, causing every single image to be seen in training. 
    # Thus, every model would eventually get 100% accuracy due to memorization, as it has already seen all the test images. 
    # Therefore the code has been slightly modified into the above to prevent the data leakage and train normally.
    # 
    # best_iou = 0.0
    # for epoch in range(args.epochs):
    #     accuracy = 0.
    #     for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    #
    #         train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    #         test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    #         trainloader = torch.utils.data.DataLoader(
    #                        dataset,
    #                        batch_size=args.batch_size, num_workers=args.num_workers,
    #                        sampler=train_subsampler)
    #         testloader = torch.utils.data.DataLoader(
    #                        dataset,
    #                        batch_size=1, num_workers=args.num_workers,
    #                        sampler=test_subsampler)
    #
    #         logging.info('Beginning Epoch {:02d}'.format(epoch))
    #         print("lr:", optimizer.state_dict()['param_groups'][0]['lr'])
    #         train_results = train(epoch, net, device, trainloader, optimizer, args.batches_per_epoch)
    #         schedule.step()  # called 5x per epoch (once per fold) — too fast
    #
    #         logging.info('Validating...')
    #         test_results = validate(net, device, testloader, args.val_batches)
    #         logging.info('%d/%d = %f' % (test_results['correct'],
    #                                      test_results['correct'] + test_results['failed'],
    #                                      test_results['correct'] / (
    #                                                  test_results['correct'] + test_results['failed'])))
    #         logging.info('Average Inference Time: {:.4f}s'.format(test_results['avg_inference_time']))
    #
    #         try:
    #             if hasattr(net, 'flops'):
    #                 total_flops = net.flops()
    #                 logging.info('FLOPs: {:.2e}'.format(total_flops))
    #         except Exception as e:
    #             logging.debug('Could not calculate FLOPs: {}'.format(str(e)))
    #
    #         iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
    #         accuracy += iou
    #         if iou > best_iou or epoch == 0 or (epoch % 50) == 0:
    #             torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
    #             best_iou = iou
    #
    #     schedule.step()  # called a 6th time at end of epoch — also too fast
    #     print("the accuracy:", accuracy / k_folds)
    # -----------------------------------------------------------------------


if __name__ == '__main__':
    run()