import logging
import time

import torch

from models.common import post_process_output
from utils.dataset_processing import evaluation


def validate(net, device, val_data, batches_per_epoch):
    net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {},
        'avg_inference_time': 0.0,
    }

    if len(val_data) == 0:
        return results

    ld = len(val_data)

    with torch.no_grad():
        batch_idx = 0
        total_inference_time = 0.0

        while batch_idx < batches_per_epoch:
            for x, y, didx, rot, zoom_factor in val_data:
                batch_idx += 1
                if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
                    break

                xc = x.to(device)
                yc = [yy.to(device) for yy in y]

                torch.cuda.synchronize()
                start_time = time.perf_counter()
                lossd = net.compute_loss(xc, yc)
                torch.cuda.synchronize()
                total_inference_time += time.perf_counter() - start_time

                loss = lossd['loss']

                results['loss'] += loss.item() / ld
                for ln, l in lossd['losses'].items():
                    if ln not in results['losses']:
                        results['losses'][ln] = 0
                    results['losses'][ln] += l.item() / ld

                q_out, ang_out, w_out = post_process_output(
                    lossd['pred']['pos'],
                    lossd['pred']['cos'],
                    lossd['pred']['sin'],
                    lossd['pred']['width'],
                )
                success = evaluation.calculate_iou_match(
                    q_out,
                    ang_out,
                    val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                    no_grasps=2,
                    grasp_width=w_out,
                )

                if success:
                    results['correct'] += 1
                else:
                    results['failed'] += 1

            if batches_per_epoch is None:
                break

        if batch_idx > 0:
            results['avg_inference_time'] = total_inference_time / batch_idx

    return results


def train(epoch, net, device, train_data, optimizer, batches_per_epoch, vis=False):
    del vis

    results = {
        'loss': 0,
        'losses': {},
    }

    net.train()

    if len(train_data) == 0:
        return results

    batch_idx = 0
    while batch_idx < batches_per_epoch:
        for x, y, _, _, _ in train_data:
            batch_idx += 1
            if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
                break

            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc)

            loss = lossd['loss']

            if batch_idx % 10 == 0:
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

            results['loss'] += loss.item()
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if batches_per_epoch is None:
            break

    if batch_idx > 0:
        results['loss'] /= batch_idx
        for ln in results['losses']:
            results['losses'][ln] /= batch_idx

    return results