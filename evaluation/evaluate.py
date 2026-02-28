import os
import cv2
import numpy as np
from pathlib import Path
import argparse


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, required=True, help='Path to predicted mask directory')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to ground-truth annotation directory')
    parser.add_argument('--ref_idx', type=str, required=True, help='Reference frame index to skip (e.g. 000)')

    args = parser.parse_args()
    return args


def main():

    args = get_arguments()
    print("Args:", args, "\n")

    class_names = sorted(os.listdir(args.gt_path))
    class_names = [c for c in class_names if ".DS" not in c]

    mIoU, mAcc = 0, 0
    count = 0

    for class_name in class_names:

        count += 1
        gt_path_class = os.path.join(args.gt_path, class_name)
        pred_path_class = args.pred_path

        gt_images = [str(p) for p in sorted(Path(gt_path_class).rglob("*.png"))]
        pred_images = [str(p) for p in sorted(Path(pred_path_class).rglob("*.png"))]

        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()

        for gt_img, pred_img in zip(gt_images, pred_images):
            if args.ref_idx in gt_img:
                continue

            gt = cv2.imread(gt_img)
            gt = np.uint8(cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY) > 0)

            pred = cv2.imread(pred_img)
            pred = np.uint8(cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY) > 0)

            intersection, union, target = intersectionAndUnion(pred, gt)
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)

        print(class_name + ',', "IoU: %.2f," % (100 * iou_class), "Acc: %.2f\n" % (100 * accuracy_class))

        mIoU += iou_class
        mAcc += accuracy_class

    print("\nmIoU: %.2f" % (100 * mIoU / count))
    print("mAcc: %.2f\n" % (100 * mAcc / count))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersectionAndUnion(output, target):
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)

    area_intersection = np.logical_and(output, target).sum()
    area_union = np.logical_or(output, target).sum()
    area_target = target.sum()

    return area_intersection, area_union, area_target


if __name__ == '__main__':
    main()
