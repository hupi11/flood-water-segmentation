import numpy as np
import torch
from torch.nn import functional as F
import os
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from show import *
from per_segment_anything import sam_model_registry, SamPredictor
from sklearn.cluster import KMeans


def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--one_data', type=str, default=None)
    parser.add_argument('--data', type=str, default='./flood_data/Tewkesbury')
    parser.add_argument('--outdir', type=str, default='persam')
    parser.add_argument('--ckpt', type=str, default='sam_vit_h_4b8939.pth')
    parser.add_argument('--ref_idx', type=str, default='000')
    parser.add_argument('--sam_type', type=str, default='vit_h')
    parser.add_argument('--num_point', type=int, default=6)

    args = parser.parse_args()
    return args


def main():

    args = get_arguments()
    print("Args:", args)

    images_path = args.data + '/Images/'
    masks_path = args.data + '/Annotations/'
    neg_masks_path = args.data + '/neg_Annotations/'
    output_path = './outputs/' + args.outdir

    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')
    
    for obj_name in os.listdir(images_path):
        if ".DS" not in obj_name:
            persam(args, obj_name, images_path, masks_path, neg_masks_path, output_path)


def persam(args, obj_name, images_path, masks_path, neg_masks_path, output_path):

    print("\n------------> Segment " + obj_name)

    # Path preparation
    ref_image_path = os.path.join(images_path, obj_name, args.ref_idx + '.jpg')
    ref_mask_path = os.path.join(masks_path, obj_name, args.ref_idx + '.png')
    ref_negmask_path = os.path.join(neg_masks_path, 'refine_mask', args.ref_idx + '.png')
    test_images_path = os.path.join(images_path, obj_name)

    output_path = os.path.join(output_path, obj_name)
    os.makedirs(output_path, exist_ok=True)

    # Load images and masks
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    print(ref_image.shape)
    ref_mask = cv2.imread(ref_mask_path)
    ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

    ref_negmask = cv2.imread(ref_negmask_path)
    ref_negmask = cv2.cvtColor(ref_negmask, cv2.COLOR_BGR2RGB)

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device)
    print("======> Load SAM")
    if args.sam_type == 'vit_h':
        sam_type, sam_ckpt = 'vit_h', 'sam_vit_h_4b8939.pth'
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    elif args.sam_type == 'vit_t':
        sam_type, sam_ckpt = 'vit_t', 'weights/mobile_sam.pt'
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
        sam.eval()

    predictor = SamPredictor(sam)

    print("======> Obtain Location Prior")
    # Image features encoding
    predictor.set_image(ref_image)
    image_feat = predictor.features.squeeze()
    print(image_feat.shape)
    ref_mask = predictor.set_image(ref_image, ref_mask)
    print(ref_mask.shape)
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)  # h, w, c
    print(ref_feat.shape)
    ref_mask = F.interpolate(ref_mask.float(), size=ref_feat.shape[0: 2], mode="bicubic")
    print(ref_mask.shape)
    ref_mask = ref_mask.squeeze()[0]
    print(ref_mask.shape)

    # Negative mask processing
    ref_negmask = predictor.set_image(ref_image, ref_negmask)
    ref_negmask = F.interpolate(ref_negmask.float(), size=ref_feat.shape[0: 2], mode="bicubic")
    ref_negmask = ref_negmask.squeeze()[0]

    # Target positive feature extraction
    target_feat1 = ref_feat[ref_mask == 128]
    target_embedding = target_feat1.mean(0).unsqueeze(0)
    target_feat1 = target_feat1 / torch.norm(target_feat1, dim=1, keepdim=True)
    target_embedding = target_embedding.unsqueeze(0)
    print("target_embedding", target_embedding.shape)
    print("归一化后的特征张量形状:", target_feat1.shape)

    # Positive KMeans clustering
    n1 = args.num_point
    pos_features = target_feat1.cpu().numpy()
    kmeans = KMeans(n_clusters=n1, init='k-means++', random_state=42)
    kmeans.fit(pos_features)

    cluster_indices1 = kmeans.labels_
    centroids1 = torch.zeros(n1, target_feat1.shape[1], device=target_feat1.device)
    for i in range(args.num_point):
        cluster_points = target_feat1[cluster_indices1 == i]
        centroids1[i] = torch.mean(cluster_points, dim=0)
        print("质心数组：", centroids1.shape)

    # Negative KMeans clustering
    target_feat2 = ref_feat[ref_negmask == 128]
    print(target_feat2.shape)
    n2 = args.num_point
    neg_features = target_feat2.cpu().numpy()
    kmeans = KMeans(n_clusters=n2, init='k-means++', random_state=43)
    kmeans.fit(neg_features)

    cluster_indices2 = kmeans.labels_
    centroids2 = torch.zeros(n2, target_feat2.shape[1], device=target_feat2.device)
    for i in range(args.num_point):
        cluster_points = target_feat2[cluster_indices2 == i]
        centroids2[i] = torch.mean(cluster_points, dim=0)
        print("质心数组：", centroids2.shape)

    print('======> Start Testing')
    if args.one_data is not None:
        # Load test image
        test_image_path = args.one_data
        filename = test_image_path.split('/')[-1]
        test_idx = filename.split('.')[0]
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

        # Image feature encoding
        predictor.set_image(test_image)
        test_feat = predictor.features.squeeze()
        print(test_feat.shape)
        C, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(C, h * w)
        topk_xy = np.array([])
        topk_label = np.array([])

        # Positive prompt selection
        oversim = []
        for i in range(args.num_point):
            print('======> 开始', i + 1, '个质心')
            centroids_feat = centroids1[i]
            sim = centroids_feat @ test_feat
            sim = sim.reshape(1, 1, h, w)
            sim = F.interpolate(sim.float(), scale_factor=4, mode="bicubic")
            sim = predictor.model.postprocess_masks(
                            sim,
                            input_size=predictor.input_size,
                            original_size=predictor.original_size).squeeze()
            print('sim', sim.shape)
            topk_xy_i, topk_label_i = pos_point_selection(sim, topk=n1, n_p=n1)
            print("正向的个数为:", len(topk_xy_i))
            if i == 0:
                topk_xy = topk_xy_i
            else:
                topk_xy = np.concatenate([topk_xy, topk_xy_i], axis=0)
            print("选取正向点的总数为:", len(topk_xy))
            topk_label = topk_label_i
            oversim.append(sim)

        # Negative prompt selection
        for i in range(args.num_point):
            print('======> 开始', i + 1, '个质心')
            centroids_feat = centroids2[i]
            sim = centroids_feat @ test_feat
            sim = sim.reshape(1, 1, h, w)
            sim = F.interpolate(sim.float(), scale_factor=4, mode="bicubic")
            sim = predictor.model.postprocess_masks(
                            sim,
                            input_size=predictor.input_size,
                            original_size=predictor.original_size).squeeze()
            print('sim', sim.shape)
            last_xy_i, last_label_i = neg_point_selection(sim, topk=n1, n_n=n2)
            print("负向的个数为:", len(last_xy_i))
            topk_xy = np.concatenate([topk_xy, last_xy_i], axis=0)
            print("选取点的总数为:", len(topk_xy))
            last_label = last_label_i

        topk_label = np.concatenate([topk_label, last_label], axis=0)

        # Obtain the target guidance for cross-attention layers
        sim_tensors = [torch.tensor(local_map) for local_map in oversim]
        stacked_sim = torch.stack(sim_tensors)
        overall_sim = torch.mean(stacked_sim, dim=0)

        overall_sim = (overall_sim - overall_sim.mean()) / torch.std(overall_sim)
        overall_sim = F.interpolate(overall_sim.unsqueeze(0).unsqueeze(0).float(), size=(64, 64), mode="bicubic")
        attn_sim = overall_sim.sigmoid_().unsqueeze(0).flatten(3)

        print("attn_sim", attn_sim.shape)
        print('topk_xy', topk_xy.shape)
        print('topk_label', topk_label)

        # First-step prediction
        masks, scores, logits, _ = predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            multimask_output=False,
            attn_sim=attn_sim,
            target_embedding=target_embedding
        )
        best_idx = np.argmax(scores)
        print(masks.shape)

        # Cascaded Post-refinement-1
        masks, scores, logits, _ = predictor.predict(
                    point_coords=topk_xy,
                    point_labels=topk_label,
                    mask_input=logits[best_idx: best_idx + 1, :, :],
                    multimask_output=True)
        best_idx = np.argmax(scores)
        print(best_idx)

        # Cascaded Post-refinement-2
        y, x = np.nonzero(masks[best_idx])
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        input_box = np.array([x_min, y_min, x_max, y_max])
        masks, scores, logits, _ = predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logits[best_idx: best_idx + 1, :, :],
            multimask_output=True)
        best_idx = np.argmax(scores)

        # Save masks
        plt.figure(figsize=(10, 10))
        plt.imshow(test_image)
        show_mask(masks[best_idx], plt.gca())
        show_points(topk_xy, topk_label, plt.gca())
        plt.title(f"Mask {best_idx}", fontsize=18)
        plt.axis('off')
        vis_mask_output_path = os.path.join(output_path, f'vis_mask_{test_idx}.jpg')
        with open(vis_mask_output_path, 'wb') as outfile:
            plt.savefig(outfile, format='jpg')

        final_mask = masks[best_idx]
        mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
        mask_colors[final_mask, :] = np.array([[0, 0, 128]])
        mask_output_path = os.path.join(output_path, test_idx + '.png')
        cv2.imwrite(mask_output_path, mask_colors)

    else:
        for test_idx in tqdm(range(len(os.listdir(test_images_path)))):

            # Load test image
            test_idx = '%03d' % test_idx
            test_image_path = test_images_path + '/' + test_idx + '.jpg'
            test_image = cv2.imread(test_image_path)
            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

            # Image feature encoding
            predictor.set_image(test_image)
            test_feat = predictor.features.squeeze()
            print(test_feat.shape)
            C, h, w = test_feat.shape
            test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
            test_feat = test_feat.reshape(C, h * w)
            topk_xy = np.array([])
            topk_label = np.array([])

            # Positive prompt selection
            for i in range(args.num_point):
                print('======> 开始', i + 1, '个质心')
                centroids_feat = centroids1[i]
                sim = centroids_feat @ test_feat
                sim = sim.reshape(1, 1, h, w)
                sim = F.interpolate(sim.float(), scale_factor=4, mode="bicubic")
                sim = predictor.model.postprocess_masks(
                                sim,
                                input_size=predictor.input_size,
                                original_size=predictor.original_size).squeeze()
                print('sim', sim.shape)
                topk_xy_i, topk_label_i = pos_point_selection(sim, topk=n1, n_p=n1)
                print("正向的个数为:", len(topk_xy_i))
                if i == 0:
                    topk_xy = topk_xy_i
                else:
                    topk_xy = np.concatenate([topk_xy, topk_xy_i], axis=0)
                print("选取正向点的总数为:", len(topk_xy))
                topk_label = topk_label_i

                oversim = []
                oversim.append(sim)

            # Negative prompt selection
            for i in range(args.num_point):
                print('======> 开始', i + 1, '个质心')
                centroids_feat = centroids2[i]
                sim = centroids_feat @ test_feat
                sim = sim.reshape(1, 1, h, w)
                sim = F.interpolate(sim.float(), scale_factor=4, mode="bicubic")
                sim = predictor.model.postprocess_masks(
                                sim,
                                input_size=predictor.input_size,
                                original_size=predictor.original_size).squeeze()
                print('sim', sim.shape)
                last_xy_i, last_label_i = neg_point_selection(sim, topk=n1, n_n=n2)
                print("负向的个数为:", len(last_xy_i))
                topk_xy = np.concatenate([topk_xy, last_xy_i], axis=0)
                print("选取点的总数为:", len(topk_xy))
                last_label = last_label_i

            topk_label = np.concatenate([topk_label, last_label], axis=0)

            # Obtain the target guidance for cross-attention layers
            sim_tensors = [torch.tensor(local_map) for local_map in oversim]
            stacked_sim = torch.stack(sim_tensors)
            overall_sim = torch.mean(stacked_sim, dim=0)
            print('overall_sim', overall_sim.shape)

            overall_sim = (overall_sim - overall_sim.mean()) / torch.std(overall_sim)
            overall_sim = F.interpolate(overall_sim.unsqueeze(0).unsqueeze(0).float(), size=(64, 64), mode="bicubic")
            attn_sim = overall_sim.sigmoid_().unsqueeze(0).flatten(3)
            print("attn_sim", attn_sim.shape)
            print('topk_xy', topk_xy.shape)
            print('topk_label', topk_label)

            # First-step prediction
            masks, scores, logits, _ = predictor.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                multimask_output=False,
                attn_sim=attn_sim,
                target_embedding=target_embedding
            )
            best_idx = np.argmax(scores)
            print(masks.shape)

            # Cascaded Post-refinement-1
            masks, scores, logits, _ = predictor.predict(
                        point_coords=topk_xy,
                        point_labels=topk_label,
                        mask_input=logits[best_idx: best_idx + 1, :, :],
                        multimask_output=True)
            best_idx = np.argmax(scores)
            print(best_idx)

            # Cascaded Post-refinement-2
            y, x = np.nonzero(masks[best_idx])
            x_min = x.min()
            x_max = x.max()
            y_min = y.min()
            y_max = y.max()
            input_box = np.array([x_min, y_min, x_max, y_max])
            masks, scores, logits, _ = predictor.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                box=input_box[None, :],
                mask_input=logits[best_idx: best_idx + 1, :, :],
                multimask_output=True)
            best_idx = np.argmax(scores)

            # Save masks
            plt.figure(figsize=(10, 10))
            plt.imshow(test_image)
            show_mask(masks[best_idx], plt.gca())
            show_points(topk_xy, topk_label, plt.gca())
            plt.title(f"Mask {best_idx}", fontsize=18)
            plt.axis('off')
            vis_mask_output_path = os.path.join(output_path, f'vis_mask_{test_idx}.jpg')
            with open(vis_mask_output_path, 'wb') as outfile:
                plt.savefig(outfile, format='jpg')

            final_mask = masks[best_idx]
            mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
            mask_colors[final_mask, :] = np.array([[0, 0, 128]])
            mask_output_path = os.path.join(output_path, test_idx + '.png')
            cv2.imwrite(mask_output_path, mask_colors)


def pos_point_selection(mask_sim, topk=1, n_p=0):
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(1)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * n_p)
    topk_xy = topk_xy.cpu().numpy()
    return topk_xy, topk_label


def neg_point_selection(mask_sim, topk=1, n_n=0):
    w, h = mask_sim.shape
    last_xy = mask_sim.flatten(0).topk(1)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = (last_xy - last_x * h)
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * n_n)
    last_xy = last_xy.cpu().numpy()
    return last_xy, last_label


if __name__ == "__main__":
    main()
