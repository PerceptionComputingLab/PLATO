import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from tqdm import tqdm
import argparse
from types import SimpleNamespace
from datetime import datetime

import torch
from torch._C import device
import torch.nn as nn
import torch.optim as optim

from metadata_manager import *
from utils.utils import *
from utils.metrics import *
from utils.loss import *
from models.model_attention import load_model
from models.PLATO import PLATO
from dataloaders import *
from logger import Logger
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

parser = argparse.ArgumentParser()
parser.add_argument(
    "--what",
    default="isic3_style_concat",
    help="Dataset to test on.",
)
parser.add_argument(
    "--load_weight_path",
    type=str,
    default="/home/Michael_Bryant/ProbModelBaseline/saved_models/LIDC/Attention_Unet+PLATO/best_model_Dice=0.701498377965981.pt",
    help="Path of the model to be tested.",
)
parser.add_argument(
    "--save_image",
    type=int,
    default=0,
    help="To save the final result or not",
)
parser.add_argument(
    "--num_filters",
    default=[32, 64, 128, 192],
    nargs="+",
    help="Number of filters per layer. Default is [32,64,128,192]",
    type=int,
)
parser.add_argument(
    "--rank",
    default=10,
    type=int,
    help="Rank for Covoriance decomposition. Default is 10",
)
parser.add_argument(
    "--sampling_times",
    default=10,
    type=int,
    help="numbers of the segmentation results sampled by SSN"
)
parser.add_argument(
    "--aleatoric_uncertainty",
    default=0,
    type=int,
    help="show aleatoric uncertainty map when =1"
)
parser.add_argument(
    "--epistemic_uncertainty",
    default=0,
    type=int,
    help="show epistemic uncertainty map when =1"
)
parser.add_argument(
    "--avg",
    default=0,
    type=int,
    help="show average among all predictions when =1"
)

def compute_hm_iou(Pred, Masks):
    lcm = np.lcm(len(Pred), len(Masks))
    len1 = len(Pred)
    len2 = len(Masks)
    for i in range((lcm // len1) - 1):
        for j in range(len1):
            Pred.append(Pred[j])
    for i in range((lcm // len2) - 1):
        for j in range(len2):
            Masks.append(Masks[j])
    #print(len(Pred))
    #print(len(Masks))
    cost_matrix = np.zeros((lcm, lcm))
    for i in range(lcm):
        for j in range(lcm):
            cost_matrix[i][j] = 1 - IoU(Pred[i], Masks[j])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    HM_IoU = np.mean([(1 - cost_matrix[i][j]) for i, j in zip(row_ind, col_ind)])
    return HM_IoU

def Dice(target, predicted_mask):
    """
    Args:
        target: (torch.tensor (batchxCxHxW)) Binary Target Segmentation from training set
        predicted_mask: (torch.tensor (batchxCxHxW)) Predicted Segmentation Mask

    Returns:
        IoU: (Float) Average IoUs over Batch
    """
    target = target.detach()
    predicted_mask = predicted_mask.detach()
    smooth = 1e-8
    true_p = (torch.logical_and(target == 1, predicted_mask == 1)).sum()
    # true_n = (torch.logical_and(target == 0, predicted_mask == 0)).sum().item() #Currently not needed for IoU
    false_p = (torch.logical_and(target == 0, predicted_mask == 1)).sum()
    false_n = (torch.logical_and(target == 1, predicted_mask == 0)).sum()
    sample_IoU = (smooth + float(true_p) + float(true_p)) / (float(true_p) + float(true_p) +
                                         float(false_p) + float(false_n) + smooth)

    return sample_IoU

def compute_max_Dice(Pred, Masks):
    len1 = len(Pred)
    len2 = len(Masks)
    mx_D = 0
    for j in range(len2):
        mx = 0
        for i in range(len1):
            Diceij = Dice(Pred[i], Masks[j])
            if Diceij > mx:
                mx = Diceij
        mx_D = mx_D + mx
    return mx_D / len2

def test(net, model, load_weight_path, test_loader, save_image, sampling_times, AU, EU, avg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(load_weight_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    sum_IoU = 0
    sum_loss = 0
    sum_Dice = 0.0
    counter = 0
    GED = 0
    DDice = 0
    print("Testing ...")
    model.eval()
    tcnt = 0
    with torch.no_grad():
        for images, masks, seg_dist, _ in tqdm(test_loader):
            tcnt = tcnt + 1
            counter += 1
            # Send tensors to cuda
            Mask = []
            images = images.to(device)
            for i in range(4):
                Mask.append(masks[i].to(device))
            #seg_dist = [x.to(device) for x in seg_dist]

            # IoU/Loss on Image Level
            # outputs logits (the mean of the distribution)
            logit, out = net(images)
            logit1, logit2, output_dict = model(logit)
            logit_distribution1 = output_dict["distribution1"]
            logit_distribution2 = output_dict["distribution2"]
            #pred_mask = (torch.sigmoid(logits)).ge(meta.masking_threshold)
            pred_mask1 = []
            pred_mask2 = []
            alpha_1 = []
            alpha_2 = []
            Avg = np.zeros((1, 128, 128))
            #sampling:
            for i in range(sampling_times):
                logits1 = logit_distribution1.sample()
                sigmoids = torch.sigmoid(logits1)
                P_mask = (torch.sigmoid(logits1)).ge(meta.masking_threshold)
                pred_mask1.append(P_mask)
                Avg = Avg + P_mask[0].cpu().numpy()
                evidence1_1 = torch.sigmoid(logits1)#Foreground Prob.
                evidence1_2 = 1 - evidence1_1#Background Prob. #[evidence1, evidence2] = logit map for 2-class softmax
                evidence1_1 = F.exp(evidence1_1)
                alpha1_1 = evidence1_1 + 1 #Foreground Uncertainty
                evidence1_2 = F.exp(evidence1_2)
                alpha1_2 = evidence1_2 + 1 #Background Uncertainty
                alpha_1.append(np.minimum(alpha1_1[0].cpu().numpy().transpose(1, 2, 0),alpha1_2[0].cpu().numpy().transpose(1, 2, 0)))
                #alpha_1[i] = scale_array(alpha_1[i])

            for i in range(sampling_times):
                logits2 = logit_distribution2.sample()
                sigmoids = torch.sigmoid(logits2)
                P_mask = (torch.sigmoid(logits2)).ge(meta.masking_threshold)
                pred_mask2.append(P_mask)
                Avg = Avg + P_mask[0].cpu().numpy()
                evidence2_1 = torch.sigmoid(logits2)#Foreground Prob.
                evidence2_2 = 1 - evidence2_1#Background Prob.#[evidence1, evidence2] = logit map for 2-class softmax
                evidence2_1 = F.exp(evidence2_1)
                alpha2_1 = evidence2_1 + 1 #Foreground Uncertainty
                evidence2_2 = F.exp(evidence2_2)
                alpha2_2 = evidence2_2 + 1 #Background Uncertainty
                alpha_2.append(np.minimum(alpha2_1[0].cpu().numpy().transpose(1, 2, 0),alpha2_2[0].cpu().numpy().transpose(1, 2, 0)))
                #alpha_2[i] = scale_array(alpha_2[i])

            if avg == 1:
                os.makedirs(f"results/{what_task}/{testing_run_name}/", exist_ok=True)
                Avg = Avg * 1.0 / (2 * sampling_times)
                plt.axis('off')
                plt.imshow(Avg.transpose(1, 2, 0),'gray')
                plt.title('Average')
                plt.savefig(f'results/{what_task}/{testing_run_name}/Average{tcnt}.png')

            if save_image + AU + EU >= 1:# Visualize the predicted result
                os.makedirs(f"results/{what_task}/{testing_run_name}/", exist_ok=True)
                plt.subplot(3, 5, 1)
                plt.axis('off')
                plt.imshow(images[0].cpu().numpy().transpose(1, 2, 0),'gray')
                plt.title('image')
                plt.subplot(3, 5, 2)
                plt.axis('off')
                plt.imshow(Mask[0][0].cpu().numpy().transpose(1, 2, 0),'gray')
                plt.title('GT1')
                plt.subplot(3, 5, 3)
                plt.axis('off')
                plt.imshow(Mask[1][0].cpu().numpy().transpose(1, 2, 0),'gray')
                plt.title('GT2')
                plt.subplot(3, 5, 4)
                plt.axis('off')
                plt.imshow(Mask[2][0].cpu().numpy().transpose(1, 2, 0),'gray')
                plt.title('GT3')
                plt.subplot(3, 5, 5)
                plt.axis('off')
                plt.imshow(Mask[3][0].cpu().numpy().transpose(1, 2, 0),'gray')
                plt.title('GT4')
                plt.subplot(3, 5, 6)
                plt.axis('off')
                plt.imshow(pred_mask1[0][0].cpu().numpy().transpose(1, 2, 0),'gray')
                plt.title('Pred1')
                plt.subplot(3, 5, 7)
                plt.axis('off')
                plt.imshow(pred_mask1[1][0].cpu().numpy().transpose(1, 2, 0),'gray')
                plt.title('Pred2')
                plt.subplot(3, 5, 8)
                plt.axis('off')
                plt.imshow(pred_mask1[2][0].cpu().numpy().transpose(1, 2, 0),'gray')
                plt.title('Pred3')
                plt.subplot(3, 5, 9)
                plt.axis('off')
                plt.imshow(pred_mask1[3][0].cpu().numpy().transpose(1, 2, 0),'gray')
                plt.title('Pred4')
                plt.subplot(3, 5, 10)
                plt.axis('off')
                plt.imshow(pred_mask1[4][0].cpu().numpy().transpose(1, 2, 0),'gray')
                plt.title('Pred5')
                plt.subplot(3, 5, 11)
                plt.axis('off')
                plt.imshow(pred_mask2[0][0].cpu().numpy().transpose(1, 2, 0),'gray')
                plt.title('Pred1')
                plt.subplot(3, 5, 12)
                plt.axis('off')
                plt.imshow(pred_mask2[1][0].cpu().numpy().transpose(1, 2, 0),'gray')
                plt.title('Pred2')
                plt.subplot(3, 5, 13)
                plt.axis('off')
                plt.imshow(pred_mask2[2][0].cpu().numpy().transpose(1, 2, 0),'gray')
                plt.title('Pred3')
                plt.subplot(3, 5, 14)
                plt.axis('off')
                plt.imshow(pred_mask2[3][0].cpu().numpy().transpose(1, 2, 0),'gray')
                plt.title('Pred4')
                plt.subplot(3, 5, 15)
                plt.axis('off')
                plt.imshow(pred_mask2[4][0].cpu().numpy().transpose(1, 2, 0),'gray')
                plt.title('Pred5')
                #plt.show()
                plt.savefig(f'results/{what_task}/{testing_run_name}/{tcnt}.png')

            if AU == True and tcnt <= 40:
                os.makedirs(f"Aleatoric_uncertainty/{what_task}/{testing_run_name}/", exist_ok=True)
                plt.subplot(2, 5, 1)
                plt.axis('off')
                plt.imshow(alpha_1[0], cmap='coolwarm', interpolation='nearest')  # 使用'coolwarm'颜色映射绘制热图
                #plt.colorbar()  # 添加颜色条
                plt.title('Uncertainty1')
                plt.subplot(2, 5, 2)
                plt.axis('off')
                plt.imshow(alpha_1[1], cmap='coolwarm', interpolation='nearest')  # 使用'coolwarm'颜色映射绘制热图
                #plt.colorbar()  # 添加颜色条
                plt.title('Uncertainty2')
                plt.subplot(2, 5, 3)
                plt.axis('off')
                plt.imshow(alpha_1[2], cmap='coolwarm', interpolation='nearest')  # 使用'coolwarm'颜色映射绘制热图
                #plt.colorbar()  # 添加颜色条
                plt.title('Uncertainty3')
                plt.subplot(2, 5, 4)
                plt.axis('off')
                plt.imshow(alpha_1[3], cmap='coolwarm', interpolation='nearest')  # 使用'coolwarm'颜色映射绘制热图
                #plt.colorbar()  # 添加颜色条
                plt.title('Uncertainty4')
                plt.subplot(2, 5, 5)
                plt.axis('off')
                plt.imshow(alpha_1[4], cmap='coolwarm', interpolation='nearest')  # 使用'coolwarm'颜色映射绘制热图
                #plt.colorbar()  # 添加颜色条
                plt.title('Uncertainty5')
                plt.subplot(2, 5, 6)
                plt.axis('off')
                plt.imshow(alpha_2[0], cmap='coolwarm', interpolation='nearest')  # 使用'coolwarm'颜色映射绘制热图
                #plt.colorbar()  # 添加颜色条
                plt.title('Uncertainty1')
                plt.subplot(2, 5, 7)
                plt.axis('off')
                plt.imshow(alpha_2[1], cmap='coolwarm', interpolation='nearest')  # 使用'coolwarm'颜色映射绘制热图
                #plt.colorbar()  # 添加颜色条
                plt.title('Uncertainty2')
                plt.subplot(2, 5, 8)
                plt.axis('off')
                plt.imshow(alpha_2[2], cmap='coolwarm', interpolation='nearest')  # 使用'coolwarm'颜色映射绘制热图
                #plt.colorbar()  # 添加颜色条
                plt.title('Uncertainty3')
                plt.subplot(2, 5, 9)
                plt.axis('off')
                plt.imshow(alpha_2[3], cmap='coolwarm', interpolation='nearest')  # 使用'coolwarm'颜色映射绘制热图
                #plt.colorbar()  # 添加颜色条
                plt.title('Uncertainty4')
                plt.subplot(2, 5, 10)
                plt.axis('off')
                plt.imshow(alpha_2[4], cmap='coolwarm', interpolation='nearest')  # 使用'coolwarm'颜色映射绘制热图
                #plt.colorbar()  # 添加颜色条
                plt.title('Uncertainty5')
                plt.savefig(f'Aleatoric_uncertainty/{what_task}/{testing_run_name}/{tcnt}.png')

            if EU == True and tcnt <= 20:
                os.makedirs(f"Epistemic_uncertainty/{what_task}/{testing_run_name}/", exist_ok=True)
                eu1 = torch.stack([pred_mask1[0], pred_mask1[1], pred_mask1[2], pred_mask1[3], pred_mask1[4]]).float()
                eu2 = torch.stack([pred_mask2[0], pred_mask2[1], pred_mask2[2], pred_mask2[3], pred_mask2[4]]).float()
                eu1 = torch.var(eu1, dim=0)
                eu2 = torch.var(eu2, dim=0)
                #print(eu1.shape)
                plt.subplot(1, 2, 1)
                plt.axis('off')
                plt.title('EU1')
                plt.imshow(eu1[0].cpu().numpy().transpose(1, 2, 0), cmap='coolwarm', interpolation='nearest')
                plt.subplot(1, 2, 2)
                plt.axis('off')
                plt.title('EU2')
                plt.imshow(eu2[0].cpu().numpy().transpose(1, 2, 0), cmap='coolwarm', interpolation='nearest')
                plt.savefig(f'Epistemic_uncertainty/{what_task}/{testing_run_name}/{tcnt}.png')

            '''
            loss_function = StochasticSegmentationNetworkLossMCIntegral(
                num_mc_samples=20
            )
            loss = loss_function(logits, masks, logit_distribution)
            sum_IoU += IoU(masks, pred_mask)
            sum_loss += loss
            '''

            #mxD = 0.0
            #mxD += compute_max_Dice(pred_mask2, Mask)
            #mxD += compute_max_Dice(pred_mask1, Mask)
            #mxD = 0.5 * mxD
            #sum_Dice += mxD

            GED1, _ = ged(Mask, pred_mask1)
            GED2, _ = ged(Mask, pred_mask2)
            GED += 0.5 * (GED1 + GED2)
            print(f'Current Mean GED = {GED/tcnt}')

            H = 0
            H += compute_hm_iou(pred_mask1, Mask)
            H += compute_hm_iou(pred_mask2, Mask)
            sum_IoU += (H / 2)
            print(f'Current Mean HM-IoU = {sum_IoU/tcnt}')

    #print(f"Test Finished! maxDice={sum_Dice/len(test_loader)}")

if __name__ == "__main__":
    # Load parsed arguments from command lind
    args = parser.parse_args()

    what_task = args.what
    load_weight_path = args.load_weight_path
    save_image = bool(args.save_image)
    num_filters = args.num_filters
    rank = args.rank
    sampling_times = args.sampling_times
    AU = args.aleatoric_uncertainty
    EU = args.epistemic_uncertainty
    avg = args.avg

    testing_run_name = (
        str(datetime.now())[:16]
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
    )
    # os.makedirs(f"results/{what_task}/{testing_run_name}/", exist_ok=True)

    meta_dict = get_meta(what_task)
    meta = SimpleNamespace(**meta_dict)

    print(f"Modelname: {testing_run_name}")
    # Check for GPU

    if torch.cuda.is_available():
        print("\nThe model will be run on GPU.")
    else:
        print("\nNo GPU available!")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nUsing the {meta.description} dataset.\n")

    # Init a model
    atten_unet = load_model(model_name='UNet').to(device)
    checkpoint = torch.load("/home/Michael_Bryant/ProbModelBaseline/saved_models/LIDC/Attention_Unet/best_model_Dice=0.6284789358861282.pt")
    atten_unet.load_state_dict(checkpoint["model_state_dict"])

    plato = PLATO(name = testing_run_name).to(device)

    if what_task=="LIDC":
        test_loader, _ = get_dataloader_2(
        task="LIDC", split="test", batch_size=1, shuffle=False, splitratio=[0.8, 0.0, 0.2], randomsplit=False
    )
        print(len(test_loader))
    else:
        test_loader, _ = get_dataloader(
            task=what_task, split="test", batch_size=1, shuffle=False, randomsplit=True
        )
        print(len(test_loader))

    # Empty GPU Cache
    torch.cuda.empty_cache()
    # StartTesting
    test(net=atten_unet, model=plato, load_weight_path=load_weight_path, test_loader=test_loader, save_image=save_image, sampling_times=sampling_times, AU=AU, EU=EU, avg=avg)


