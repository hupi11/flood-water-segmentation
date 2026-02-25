import numpy as np
import torch
from torch.nn import functional as F
import sys
import os
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')
from sklearn.manifold import TSNE
from show import *
from per_segment_anything import sam_model_registry, SamPredictor
from sklearn.cluster import KMeans
from matplotlib.markers import MarkerStyle
import seaborn as sns
from sklearn.decomposition import PCA

def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--one_data', type=str, default=None)
    parser.add_argument('--data', type=str, default='./flood_data/Tewkesbury')
    parser.add_argument('--outdir', type=str, default='persam')
    parser.add_argument('--ckpt', type=str, default='sam_vit_h_4b8939.pth')
    parser.add_argument('--ref_idx', type=str, default='000')
    parser.add_argument('--ref_idx1', type=str, default='009')
    parser.add_argument('--ref_idx2', type=str, default='014')
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
            persam(args, obj_name, images_path, masks_path,neg_masks_path, output_path)



def draw_star(ax, x, y, size, color):
    star = MarkerStyle(marker=(5, 1, 0))
    ax.scatter(x, y, marker=star, s=size, color=color, edgecolor='k')

def persam(args, obj_name, images_path, masks_path,neg_masks_path, output_path):  
    # 注释 obj_name==[water,bag]    images_path==/Personalize-SAM/data2/Images  

    print("\n------------> Segment " + obj_name)
    
    # Path preparation

    ref_image_path = os.path.join(images_path, obj_name, args.ref_idx + '.jpg')  #ref_image_path==water中序号为00的image
    ref_mask_path = os.path.join(masks_path, obj_name, args.ref_idx + '.png')
    ref_negmask_path = os.path.join(neg_masks_path, 'refine_mask', args.ref_idx + '.png')

    ref_mask_path1 = os.path.join(masks_path, obj_name, args.ref_idx1 + '.png')

    ref_mask_path2 = os.path.join(masks_path, obj_name, args.ref_idx2 + '.png')

    test_images_path = os.path.join(images_path, obj_name) #test_image_path==water中所有images

    output_path = os.path.join(output_path, obj_name)
    os.makedirs(output_path, exist_ok=True)

    # Load images and masks
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB) #将图片通道改为RGB 维度0，1，2换位置为 2，1，0
    print(ref_image.shape)
    ref_mask = cv2.imread(ref_mask_path)
    ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

    ref_negmask = cv2.imread(ref_negmask_path)
    ref_negmask = cv2.cvtColor(ref_negmask, cv2.COLOR_BGR2RGB)

    # ref_mask1 = cv2.imread(ref_mask_path1)
    # ref_mask1 = cv2.cvtColor(ref_mask1, cv2.COLOR_BGR2RGB)
    # ref_mask2 = cv2.imread(ref_mask_path2)
    # ref_mask2 = cv2.cvtColor(ref_mask2, cv2.COLOR_BGR2RGB)
    # print(ref_mask.shape)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device)
    print("======> Load SAM" )
    if args.sam_type == 'vit_h':
        sam_type, sam_ckpt = 'vit_h', 'sam_vit_h_4b8939.pth'
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    elif args.sam_type == 'vit_t':
        sam_type, sam_ckpt = 'vit_t', 'weights/mobile_sam.pt'
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
        sam.eval()

    predictor = SamPredictor(sam)

    print("======> Obtain Location Prior" )
    # Image features encoding
    predictor.set_image(ref_image)
    image_feat = predictor.features.squeeze() #c,h,w
    print(image_feat.shape)
    ref_mask = predictor.set_image(ref_image, ref_mask) #torch.Size([1, 3, 1024, 1024])
    print(ref_mask.shape)
    ref_feat = predictor.features.squeeze().permute(1, 2, 0) #h,w,c ！！！！！！！Fr输入图像的特征 torch.Size([64, 64, 256])
    print(ref_feat.shape)
    ref_mask = F.interpolate(ref_mask.float(), size=ref_feat.shape[0: 2], mode="bicubic") #torch.Size([1, 3, 64, 64]) c,h,w
    print(ref_mask.shape)
    ref_mask = ref_mask.squeeze()[0] #torch.Size([64, 64])
    print(ref_mask.shape)

    #neg_mask 
    ref_negmask = predictor.set_image(ref_image, ref_negmask) #torch.Size([1, 3, 1024, 1024])
    ref_negmask = F.interpolate(ref_negmask.float(), size=ref_feat.shape[0: 2], mode="bicubic")
    ref_negmask = ref_negmask.squeeze()[0]



    #####################输出mask
    # import torch
    # # 假设 ref_mask 是一个 PyTorch 张量
    # min_value = torch.min(ref_negmask)
    # max_value = torch.max(ref_negmask)

    # print("Minimum value in ref_mask:", min_value.item())
    # print("Maximum value in ref_mask:", max_value.item())
    # ref_negmask=ref_negmask.cpu()
    # ref_negmask=ref_negmask.numpy()
    # # Get unique values in the mask and their counts
    # unique_values, counts = np.unique(ref_negmask, return_counts=True)

    # # Print unique values and their counts
    # for value, count in zip(unique_values, counts):
    #     print(f"Value: {value}, Count: {count}")
    # sys.exit()
    # ref_mask1 = predictor.set_image(ref_image, ref_mask1) #torch.Size([1, 3, 1024, 1024])
    # ref_mask1 = F.interpolate(ref_mask1.float(), size=ref_feat.shape[0: 2], mode="bicubic") #torch.Size([1, 3, 64, 64]) c,h,w
    # ref_mask1 = ref_mask1.squeeze()[0] #torch.Size([64, 64])
    # print(ref_mask1.shape)
    # ref_mask2 = predictor.set_image(ref_image, ref_mask2) #torch.Size([1, 3, 1024, 1024])
    # ref_mask2 = F.interpolate(ref_mask2.float(), size=ref_feat.shape[0: 2], mode="bicubic") #torch.Size([1, 3, 64, 64]) c,h,w
    # ref_mask2 = ref_mask2.squeeze()[0] #torch.Size([64, 64])
    # print(ref_mask2.shape)

    # #Target positive feature extraction
    # target_feat_p1 = ref_feat[ref_mask ==128] #选取经过sam后的mask中大于0的部分作为目标特征 torch.Size([263, 256]) 263 表示选取的目标特征的数量，256 表示每个特征的维度大小。
    # print(target_feat_p1.shape)
    # target_feat_p2 = ref_feat[ref_mask1 == 128]
    # print(target_feat_p2.shape)
    # target_feat_p3 = ref_feat[ref_mask2 == 128]
    # print(target_feat_p3.shape)
    # # 提取特征向量中的索引
    # combined_features = torch.cat((target_feat_p1, target_feat_p2, target_feat_p3), dim=0)
    # # 提取共有特征
    # target_feat1 = torch.unique(combined_features, dim=0)
    # print(target_feat1.shape)
    target_feat1 = ref_feat[ref_mask == 128] #选取经过sam后的mask中大于0的部分作为目标特征
    target_embedding = target_feat1.mean(0).unsqueeze(0)
    # target_feat1_mean = target_feat1.mean(0)
    # target_feat1_max = torch.max(target_feat1, dim=0)[0]
    # target_feat1 = (target_feat1_max / 2 + target_feat1_mean / 2).unsqueeze(0)
    target_feat1 = target_feat1 / torch.norm(target_feat1, dim=1, keepdim=True)#归一化
    target_embedding = target_embedding.unsqueeze(0) #获取到的目标特征张量
    #target_embedding = target_feat1.mean(0).unsqueeze(0)
    # target_feat1 = target_feat1 / torch.norm(target_feat1, dim=1, keepdim=True)#归一化
    # target_embedding = target_embedding.unsqueeze(0) #embedding 供后面使用 ????????????       
    print("target_embedding",target_embedding.shape) 
    # 确保归一化后的特征张量的形状
    print("归一化后的特征张量形状:", target_feat1.shape)


###############################################################
    #聚类个数
    n1 = args.num_point
    
    pos_features=target_feat1.cpu().numpy()

    # 使用 KMeans++ 算法将特征向量 F 分为 4 类
    kmeans = KMeans(n_clusters=n1, init='k-means++', random_state=42)  #
    kmeans.fit(pos_features)


##############################    # ##########画图
    # cluster_indices=kmeans.fit(pos_features)
    # centroids = kmeans.cluster_centers_
    # #可视化聚类
    # save_path = 'clustering_visualization1.png'  # 指定保存路径
    
    # # 获取聚类标签
    # cluster_labels = kmeans.predict(pos_features)

    # # 使用 PCA 进行降维
    # pca = PCA(n_components=2)
    # pos_features_2d = pca.fit_transform(pos_features)
    # centroid_2d = pca.transform(centroids)

    # # 柔和的颜色列表
    # palette = sns.color_palette("autumn", n_colors=n1)

    # # 可视化聚类结果
    # plt.figure(figsize=(10, 8))
    # ax = plt.gca()

    # for cluster_label in range(n1):
    #     ax.scatter(pos_features_2d[cluster_labels == cluster_label, 0],
    #             pos_features_2d[cluster_labels == cluster_label, 1],
    #             color=palette[cluster_label % len(palette)],  # 使用柔和的颜色
    #             label=f'Cluster {cluster_label + 1}')

    # # 绘制质心
    # for cluster_label in range(n1):
    #     star = MarkerStyle(marker=(5, 1, 0))
    #     ax.scatter(centroid_2d[cluster_label, 0], centroid_2d[cluster_label, 1], 
    #             marker=star, s=1200, color=palette[cluster_label % len(palette)], edgecolor='k', label=f'Centroid {cluster_label + 1}')  # 五角星标记
    # plt.title('pos_Clustering Visualization')
    # plt.xlabel('Feature Dimension 1')
    # plt.ylabel('Feature Dimension 2')
    # #plt.legend()
    # plt.grid(False)

    # # 保存可视化结果
    # plt.savefig(save_path,dpi=300)
    # plt.close()






    # 获取每个样本的簇标签
    cluster_indices1 = kmeans.labels_

    # 获取每个部分的质心
    centroids1 = torch.zeros(n1, target_feat1.shape[1],device=target_feat1.device)
    for i in range(args.num_point):
        cluster_points = target_feat1[cluster_indices1 == i]
        centroids1[i] = torch.mean(cluster_points, dim=0)
        print("质心数组：", centroids1.shape)    

    


    # Target positive feature extraction
    target_feat2 = ref_feat[ref_negmask == 128]
    print(target_feat2.shape)
        #聚类个数
    n2 = args.num_point #args.num_point
    neg_features=target_feat2.cpu().numpy()
    # 使用 KMeans++ 算法将特征向量 F 分为 4 类
    kmeans = KMeans(n_clusters=n2, init='k-means++', random_state=43)
    kmeans.fit(neg_features)
    
    # #####下面是聚类画图
    # cluster_indices=kmeans.fit(neg_features)
    # centroids = kmeans.cluster_centers_
    # #可视化聚类
    # save_path = 'neg_clustering_visualization.png'  # 指定保存路径
    
    # # 获取聚类标签
    # cluster_labels = kmeans.predict(neg_features)

    # # 使用 PCA 进行降维
    # pca = PCA(n_components=2)
    # neg_features_2d = pca.fit_transform(neg_features)
    # centroid_2d = pca.transform(centroids)

    # # 柔和的颜色列表
    # palette = sns.color_palette("cool", n_colors=n1)

    # # 可视化聚类结果
    # plt.figure(figsize=(10, 8))
    # ax = plt.gca()

    # for cluster_label in range(n1):
    #     ax.scatter(neg_features_2d[cluster_labels == cluster_label, 0],
    #             neg_features_2d[cluster_labels == cluster_label, 1],
    #             color=palette[cluster_label % len(palette)],  # 使用柔和的颜色
    #             label=f'Cluster {cluster_label + 1}')

    # # 绘制质心
    # for cluster_label in range(n1):
    #     star = MarkerStyle(marker=(5, 1, 0))
    #     ax.scatter(centroid_2d[cluster_label, 0], centroid_2d[cluster_label, 1], 
    #             marker=star, s=1200, color=palette[cluster_label % len(palette)], edgecolor='k', label=f'Centroid {cluster_label + 1}')  # 五角星标记
    # plt.title('Clustering Visualization')
    # plt.xlabel('Feature Dimension 1')
    # plt.ylabel('Feature Dimension 2')
    # #plt.legend()
    # plt.grid(False)

    # # 保存可视化结果
    # plt.savefig(save_path,dpi=300)
    # plt.close()
    # sys.exit()


















    # 获取每个样本的簇标签
    cluster_indices2 = kmeans.labels_

    # 获取每个部分的质心
    centroids2 = torch.zeros(n2, target_feat2.shape[1],device=target_feat2.device)
    for i in range(args.num_point):
        cluster_points = target_feat2[cluster_indices2 == i]
        centroids2[i] = torch.mean(cluster_points, dim=0)
        print("质心数组：", centroids2.shape)    



###############################################
    print('======> Start Testing')
    if args.one_data!=None:
        # Load test image
        test_image_path = args.one_data #/Personalize-SAM/data2/Images/water/01.jpg 路径，所以images命名必须00，01
        filename = test_image_path.split('/')[-1]  # This will get '001.jpg'
        # Remove the file extension to isolate '001'
        test_idx = filename.split('.')[0]  # This will get '001'
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

        # Image feature encoding
        predictor.set_image(test_image) #此处没有mask
        test_feat = predictor.features.squeeze()
        print(test_feat.shape)
        # Cosine similarity
        C, h, w = test_feat.shape   #其中C是特征的通道数，h和w是特征的高度和宽度。
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True) #归一化
        test_feat = test_feat.reshape(C, h * w)  #这行代码将特征张量展平成一个向量，这样每个像素点的特征就变成了向量中的一个元素。
        topk_xy = np.array([])
        topk_label = np.array([])
        #positive
        oversim =[] #用来存储sim，后面进行平均池化来得到overall sim
        negoversim=[]
        for i in range(args.num_point):
            print('======> 开始',i+1,'个质心')
            centroids_feat = centroids1[i]
            sim = centroids_feat @ test_feat  #这行代码计算了目标特征向量（target_feat）和测试特征向量（test_feat）之间的相似度。@符号表示了矩阵乘法操作。

            sim = sim.reshape(1, 1, h, w)  #相似度张量
            sim = F.interpolate(sim.float(), scale_factor=4, mode="bicubic")
            sim = predictor.model.postprocess_masks(
                            sim,
                            input_size=predictor.input_size,
                            original_size=predictor.original_size).squeeze()
            print('sim',sim.shape)

            # Positive location prior 其中topk等于聚类个数
            topk_xy_i, topk_label_i = pos_point_selection(sim, topk=n1,n_p=n1)
            topk_xy_i_count = len(topk_xy_i)
            print("正向的个数为:", topk_xy_i_count)
            if i==0:
                topk_xy = topk_xy_i
            else:
                topk_xy = np.concatenate([topk_xy,topk_xy_i], axis=0)
            topk_xy_count = len(topk_xy)
            print("选取正向点的总数为:", topk_xy_count)
            topk_label=topk_label_i

            
            oversim.append(sim)
        last_xy = np.array([])
        last_label = np.array([])
        #negitive
        for i in range(args.num_point):
            print('======> 开始',i+1,'个质心')
            centroids_feat = centroids2[i]
            sim = centroids_feat @ test_feat  #这行代码计算了目标特征向量（target_feat）和测试特征向量（test_feat）之间的相似度。@符号表示了矩阵乘法操作。

            sim = sim.reshape(1, 1, h, w)  #相似度张量
            sim = F.interpolate(sim.float(), scale_factor=4, mode="bicubic")
            sim = predictor.model.postprocess_masks(
                            sim,
                            input_size=predictor.input_size,
                            original_size=predictor.original_size).squeeze()
            print('sim',sim.shape)

            # Negitive location prior 其中topk等于聚类个数
            last_xy_i, last_label_i = neg_point_selection(sim, topk=n1,n_n=n2)
            last_xy_i_count = len(last_xy_i)
            print("负向的个数为:", last_xy_i_count)

            topk_xy = np.concatenate([topk_xy,last_xy_i], axis=0)
            topk_xy_count = len(topk_xy)
            print("选取点的总数为:", topk_xy_count)
            last_label=last_label_i
            negoversim.append(sim)

        topk_label=np.concatenate([topk_label,last_label], axis=0)
        print(f"over_sim shape: {len(oversim)}")


#####################        # # Obtain the target guidance for cross-attention layers
        # ##########正向置信图
        # sim_tensors = [torch.tensor(local_map) for local_map in oversim]
        # # 将局部地图堆叠成一个张量，维度为 (n, height, width)
        # stacked_sim = torch.stack(sim_tensors)
        
        # ################################单个置信图
        # # 确保张量在CPU上并转换为numpy数组
        # stacked_sim1 = stacked_sim.detach().cpu().numpy()
        # print(f"stacked_sim shape: {stacked_sim1.shape}")

        # # 绘制每个单独的置信图
        # for i, sim_map in enumerate(stacked_sim1):
        #     plt.figure(figsize=(10, 8))
        #     plt.imshow(sim_map, cmap='YlGnBu')  # 你可以选择其他颜色映射，如 'jet' 或 'YlGn'
        #     plt.colorbar()
        #     plt.title(f"Confidence Map {i+1}")
        #     plt.savefig(f"confidence_map_{i+1}.png",dpi=300)
        #     plt.close()

        # ##########负向置信图
        # neg_sim_tensors = [torch.tensor(local_map) for local_map in negoversim]
        # # 将局部地图堆叠成一个张量，维度为 (n, height, width)
        # neg_stacked_sim = torch.stack(neg_sim_tensors)
        
        # ################################单个置信图
        # # 确保张量在CPU上并转换为numpy数组
        # neg_stacked_sim1 = neg_stacked_sim.detach().cpu().numpy()
        # print(f"neg_stacked_sim shape: {neg_stacked_sim1.shape}")

        # # 绘制每个单独的置信图
        # for i, sim_map in enumerate(neg_stacked_sim1):
        #     plt.figure(figsize=(10, 8))
        #     plt.imshow(sim_map, cmap='Oranges')  # 你可以选择其他颜色映射，如 'jet' 或 'YlGn'
        #     plt.colorbar()
        #     plt.title(f"neg_Confidence Map {i+1}")
        #     plt.savefig(f"neg_confidence_map_{i+1}.png",dpi=300)
        #     plt.close()


        
        overall_sim = torch.mean(stacked_sim, dim=0)
        # overall_sim = (overall_sim - overall_sim.mean()) / torch.std(overall_sim)
        # print('overall_sim',overall_sim.shape)
        # overall_sim = overall_sim.detach().cpu().numpy()
        # # 绘制置信图
        # # 绘制置信图并保存为图像文件

        # # 定义自定义颜色映射
        # cdict = {
        #     'red':   [(0.0,  1.0, 1.0),
        #             (0.5,  1.0, 1.0),
        #             (1.0,  0.0, 0.0)],

        #     'green': [(0.0,  1.0, 1.0),
        #             (0.5,  1.0, 1.0),
        #             (1.0,  0.0, 0.0)],

        #     'blue':  [(0.0,  1.0, 1.0),
        #             (0.5,  0.0, 0.0),
        #             (1.0,  0.0, 0.0)]
        # }

        # custom_cmap = mcolors.LinearSegmentedColormap('custom_cmap', cdict)


        # plt.figure(figsize=(6, 6))
        # plt.imshow(overall_sim, cmap='YlGn')
        # plt.colorbar()
        # plt.title("Overall Confidence Map")
        # plt.savefig("overall_confidence_map_YlGn.png",dpi=300)
        # plt.close()
        


        overall_sim = (overall_sim - overall_sim.mean()) / torch.std(overall_sim)
        overall_sim = F.interpolate(overall_sim.unsqueeze(0).unsqueeze(0).float(), size=(64, 64), mode="bicubic")
        attn_sim = overall_sim.sigmoid_().unsqueeze(0).flatten(3)


        print("attn_sim",attn_sim.shape)
        print('topk_xy',topk_xy.shape)
        print('topk_label',topk_label)
        # First-step prediction
        masks, scores, logits, _ = predictor.predict(
            point_coords=topk_xy, 
            point_labels=topk_label, 
            multimask_output=False,
            attn_sim=attn_sim,  # Target-guided Attention
            target_embedding=target_embedding  # Target-semantic Prompting
        )
        best_idx = np.argmax(scores) #0
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
            test_idx = '%03d' % test_idx #格式化字符串如果idx为1，那么被格式化为01
            test_image_path = test_images_path + '/' + test_idx + '.jpg' #/Personalize-SAM/data2/Images/water/01.jpg 路径，所以images命名必须00，01
            test_image = cv2.imread(test_image_path)
            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

            # Image feature encoding
            predictor.set_image(test_image) #此处没有mask
            test_feat = predictor.features.squeeze()
            print(test_feat.shape)
            # Cosine similarity
            C, h, w = test_feat.shape   #其中C是特征的通道数，h和w是特征的高度和宽度。
            test_feat = test_feat / test_feat.norm(dim=0, keepdim=True) #归一化
            test_feat = test_feat.reshape(C, h * w)  #这行代码将特征张量展平成一个向量，这样每个像素点的特征就变成了向量中的一个元素。
            topk_xy = np.array([])
            topk_label = np.array([])
            #positive
            for i in range(args.num_point):
                print('======> 开始',i+1,'个质心')
                centroids_feat = centroids1[i]
                sim = centroids_feat @ test_feat  #这行代码计算了目标特征向量（target_feat）和测试特征向量（test_feat）之间的相似度。@符号表示了矩阵乘法操作。

                sim = sim.reshape(1, 1, h, w)  #相似度张量
                sim = F.interpolate(sim.float(), scale_factor=4, mode="bicubic")
                sim = predictor.model.postprocess_masks(
                                sim,
                                input_size=predictor.input_size,
                                original_size=predictor.original_size).squeeze()
                print('sim',sim.shape)

                # Positive location prior 其中topk等于聚类个数
                topk_xy_i, topk_label_i = pos_point_selection(sim, topk=n1,n_p=n1)
                topk_xy_i_count = len(topk_xy_i)
                print("正向的个数为:", topk_xy_i_count)
                if i==0:
                    topk_xy = topk_xy_i
                else:
                    topk_xy = np.concatenate([topk_xy,topk_xy_i], axis=0)
                topk_xy_count = len(topk_xy)
                print("选取正向点的总数为:", topk_xy_count)
                topk_label=topk_label_i

                oversim =[] #用来存储sim，后面进行平均池化来得到overall sim
                oversim.append(sim)
            last_xy = np.array([])
            last_label = np.array([])
            #negitive
            for i in range(args.num_point):
                print('======> 开始',i+1,'个质心')
                centroids_feat = centroids2[i]
                sim = centroids_feat @ test_feat  #这行代码计算了目标特征向量（target_feat）和测试特征向量（test_feat）之间的相似度。@符号表示了矩阵乘法操作。

                sim = sim.reshape(1, 1, h, w)  #相似度张量
                sim = F.interpolate(sim.float(), scale_factor=4, mode="bicubic")
                sim = predictor.model.postprocess_masks(
                                sim,
                                input_size=predictor.input_size,
                                original_size=predictor.original_size).squeeze()
                print('sim',sim.shape)

                # Negitive location prior 其中topk等于聚类个数
                last_xy_i, last_label_i = neg_point_selection(sim, topk=n1,n_n=n2)
                last_xy_i_count = len(last_xy_i)
                print("负向的个数为:", last_xy_i_count)

                topk_xy = np.concatenate([topk_xy,last_xy_i], axis=0)
                topk_xy_count = len(topk_xy)
                print("选取点的总数为:", topk_xy_count)
                last_label=last_label_i

            topk_label=np.concatenate([topk_label,last_label], axis=0)

            # Obtain the target guidance for cross-attention layers
            sim_tensors = [torch.tensor(local_map) for local_map in oversim]
            # 将局部地图堆叠成一个张量，维度为 (n, height, width)
            stacked_sim = torch.stack(sim_tensors)
            
            # 在第一个维度上求平均值，得到总体置信度图
            overall_sim = torch.mean(stacked_sim, dim=0)
            print('overall_sim',overall_sim.shape)
        
            overall_sim = (overall_sim - overall_sim.mean()) / torch.std(overall_sim)
            overall_sim = F.interpolate(overall_sim.unsqueeze(0).unsqueeze(0).float(), size=(64, 64), mode="bicubic")
            attn_sim = overall_sim.sigmoid_().unsqueeze(0).flatten(3)
            print("attn_sim",attn_sim.shape)
            print('topk_xy',topk_xy.shape)
            print('topk_label',topk_label)
            # First-step prediction
            masks, scores, logits, _ = predictor.predict(
                point_coords=topk_xy, 
                point_labels=topk_label, 
                multimask_output=False,
                attn_sim=attn_sim,  # Target-guided Attention
                target_embedding=target_embedding  # Target-semantic Prompting
            )
            best_idx = np.argmax(scores) #0
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
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(1)[1]    #使用 mask_sim.flatten(0).topk(topk)[1] 获取了相似度张量展平后的前 topk 个最大值的索引，这样可以找到对应的像素点。       
    topk_x = (topk_xy // h).unsqueeze(0)#将这些索引转换为 (x, y) 形式的坐标，其中 x 是索引除以高度 h 的商，y 是索引减去 x * h。
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * n_p)
    topk_xy = topk_xy.cpu().numpy()       
    
    return topk_xy, topk_label

def neg_point_selection(mask_sim, topk=1, n_n=0):
    # Last-1 point selection
    w, h = mask_sim.shape
    last_xy = mask_sim.flatten(0).topk(1)[1]    #使用 mask_sim.flatten(0).topk(topk)[1] 获取了相似度张量展平后的前 topk 个最大值的索引，这样可以找到对应的像素点。       
    lastshold_index2 = int(mask_sim.numel() * (0.30))
    last_x = (last_xy // h).unsqueeze(0)
    last_y = (last_xy - last_x * h)
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * n_n)
    last_xy = last_xy.cpu().numpy()     
    
    return last_xy, last_label

def point_mutiselection(mask_sim, topk=1):
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(1)[1]    #使用 mask_sim.flatten(0).topk(topk)[1] 获取了相似度张量展平后的前 topk 个最大值的索引，这样可以找到对应的像素点。       
    print("topkxy",topk_xy.shape)
    # 获取排序后的索引
    sorted_indices = mask_sim.flatten().argsort()
    
    # 获取排在前 5% 的一个点的索引
    threshold_index = int(mask_sim.numel() * (1 - 0.001))
    top_percentage_indices = sorted_indices[threshold_index]

    #print(top_percentage_indices)



    #print("top",top_percentage_indices.shape)
    # 将最高点和排在前 5% 的点的索引合并到 topk_xy 中
    top_percentage_index_tensor = torch.tensor(top_percentage_indices, device=mask_sim.device).unsqueeze(0)
    topk_xy = torch.cat((topk_xy, top_percentage_index_tensor), dim=0)
    
    # 获取排在前 10% 的一个点的索引
    threshold_index2 = int(mask_sim.numel() * (1 - 0.002))
    top_percentage_indices2 = sorted_indices[threshold_index2]
    top_percentage_index_tensor2 = torch.tensor(top_percentage_indices2, device=mask_sim.device).unsqueeze(0)
    topk_xy = torch.cat((topk_xy, top_percentage_index_tensor2), dim=0)

    topk_x = (topk_xy // h).unsqueeze(0)#将这些索引转换为 (x, y) 形式的坐标，其中 x 是索引除以高度 h 的商，y 是索引减去 x * h。
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()


    # Top-last point selection

    last_xy = mask_sim.flatten(0).topk(1, largest=False)[1]

    lastshold_index = int(mask_sim.numel() * (0.05))
    last_percentage_indices = sorted_indices[lastshold_index]
    
    
    last_percentage_index_tensor = torch.tensor(last_percentage_indices, device=mask_sim.device).unsqueeze(0)
    last_xy = torch.cat((last_xy, last_percentage_index_tensor), dim=0)

    lastshold_index2 = int(mask_sim.numel() * (0.30))
    last_percentage_indices2 = sorted_indices[lastshold_index2]
    last_percentage_index_tensor2 = torch.tensor(last_percentage_indices2, device=mask_sim.device).unsqueeze(0)
    last_xy = torch.cat((last_xy, last_percentage_index_tensor2), dim=0)

    last_x = (last_xy // h).unsqueeze(0)
    last_y = (last_xy - last_x * h)
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()
    
    return topk_xy, topk_label, last_xy, last_label

if __name__ == "__main__":
    main()
