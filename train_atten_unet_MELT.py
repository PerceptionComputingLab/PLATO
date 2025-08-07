import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from tqdm import tqdm
import argparse
from types import SimpleNamespace
from datetime import datetime
from utils.loss import *
import matplotlib.pyplot as plt

import torch
from torch._C import device
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from metadata_manager import *
from utils.utils import *
from utils.metrics import *
from models.model_attention import load_model
from models.PLATO import PLATO
from dataloaders import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--what",
    default="isic3_style_concat",
    help="Dataset to train on.",
)
parser.add_argument(
    "--lr",
    default=0.0001,
    type=float,
    help="Learning Rate for Training. Default is 0.0001",
)
parser.add_argument(
    "--rank",
    default=10,
    type=int,
    help="Rank for Covoriance decomposition. Default is 10",
)
parser.add_argument(
    "--epochs", default=200, type=int, help="Number of Epochs to train. Default is 200"
)
parser.add_argument(
    "--batchsize", default=6, type=int, help="Number of Samples per Batch. Default is 6"
)
parser.add_argument(
    "--weightdecay",
    default=1e-4,
    type=float,
    help="Parameter for Weight Decay. Default is 1e-4",
)
parser.add_argument(
    "--resume_epoch",
    default=0,
    type=int,
    help="Resume training at the specified epoch. Default is 0",
)
parser.add_argument(
    "--save_model",
    default=False,
    type=bool,
    help="Set True if checkpoints should be saved. Default is False",
)
parser.add_argument(
    "--testit",
    default=False,
    type=bool,
    help="Set True testing the trained model on the testset. Default is False",
)
parser.add_argument(
    "--test_threshold",
    default=0.5,
    type=float,
    help="Treshold for masking the logid/sigmoid predictions. Only use with --testit. Default is 0.5",
)
parser.add_argument(
    "--N", default=16, type=int, help="Number of Samples for GED Metric. Default is 16"
)
parser.add_argument(
    "--W",
    default=1,
    type=int,
    help="Set 0 to turn off Weights and Biases. Default is 1 (tracking)",
)
parser.add_argument(
    "--transfer",
    default="None",
    help="Activates transfer learning when given a model's name. Default is None (no transfer learning)",
)
parser.add_argument(
    '--log_dir',
    default='loggers',
    help='Store logs in this directory during training.',
    type=str
)
parser.add_argument(
    '--save_model_step',
    type=int,
    default=50
)
parser.add_argument(
    '--write',
    help='Saves the training logs',
    dest='write',
    action='store_true'
)
parser.set_defaults(
    write=True
)
parser.add_argument(
    "--num_filters",
    default=[32, 64, 128, 192],
    nargs="+",
    help="Number of filters per layer. Default is [32,64,128,192]",
    type=int,
)

def Dice_loss(a, b):
    a = a.detach()
    b = b.detach()
    true_p = (torch.logical_and(a == 1, b == 1)).sum()
    # true_n = (torch.logical_and(target == 0, predicted_mask == 0)).sum().item() #Currently not needed for IoU
    false_p = (torch.logical_and(a == 0, b == 1)).sum()
    false_n = (torch.logical_and(a == 1, b == 0)).sum()
    sample_Dice = (1e-5 + float(true_p) + float(true_p)) / (float(true_p) + float(true_p) +
                                         float(false_p) + float(false_n) + 1e-5)

    return 1 - sample_Dice

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

def train(
    net,
    model,
    resume_epoch,
    epochs,
    opt,
    train_loader,
    val_loader,
    save_checkpoints,
    transfer_model,
    metadata,
    forward_passes,
    log_dir,
    save_model_step,
    write,
    W=True
):
    # Set device to Cuda if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #logger=Logger(log_dir,write,save_model_step)
    # Check if want to resume prior checkpoints
    print(f"Training from scratch...\n")
    iterations = 0
    min_Dice = 0.0
    for epoch in range(resume_epoch, epochs):  # may be error in range
        print(f"Epochs:{epoch+1}/{epochs} ... ")
        print("Training")
        sum_batch_loss = 0
        sum_batch_GED = 0
        counter = 0
        loss = 0
        model.train()
        for images, masks, _, _ in tqdm(train_loader):
            counter += 1
            iterations += 1
            # Send tensors to Cuda
            images = images.to(device)
            Mask = []
            Mask1 = []
            Mask2 = []
            images = images.to(device)
            for i in range(4):
                Mask.append(masks[i].to(device))
            opt.zero_grad()
            # Forward pass
            logit, out = net(images)  # outputs logits
            logits1, logits2, output_dict = model(logit)
            logit_distribution1 = output_dict["distribution1"]
            logit_distribution2 = output_dict["distribution2"]
            # Treshold (default 0.5)

            #Prediction Branch
            for i in range(forward_passes):
                logits11 = logit_distribution1.sample()
                sigmoids = torch.sigmoid(logits11)
                P_mask = (torch.sigmoid(logits11)).ge(meta.masking_threshold)
                Mask1.append(P_mask)
            for i in range(forward_passes):
                logits22 = logit_distribution2.sample()
                sigmoids = torch.sigmoid(logits22)
                P_mask = (torch.sigmoid(logits22)).ge(meta.masking_threshold)
                Mask2.append(P_mask)

            #Evidence Branch
            evidence1_1 = torch.sigmoid(logits1)#Foreground Prob.
            evidence1_2 = 1 - evidence1_1#Background Prob.
            #[evidence1, evidence2] = logit map for 2-class softmax
            evidence1_1 = F.exp(evidence1_1)
            alpha1_1 = evidence1_1 + 1 #Foreground Uncertainty
            evidence1_2 = F.exp(evidence1_2)
            alpha1_2 = evidence1_2 + 1 #Background Uncertainty

            evidence2_1 = torch.sigmoid(logits2)#Foreground Prob.
            evidence2_2 = 1 - evidence2_1#Background Prob.
            #[evidence1, evidence2] = logit map for 2-class softmax
            evidence2_1 = F.exp(evidence2_1)
            alpha2_1 = evidence2_1 + 1 #Foreground Uncertainty
            evidence2_2 = F.exp(evidence2_2)
            alpha2_2 = evidence2_2 + 1 #Background Uncertainty

            loss_function = StochasticSegmentationNetworkLossMCIntegral(
                num_mc_samples=20
            )
            loss1 = 0.25*(loss_function(logits1, Mask[0], logit_distribution1)+loss_function(logits1, Mask[1], logit_distribution1)+loss_function(logits1, Mask[2], logit_distribution1)+loss_function(logits1, Mask[3], logit_distribution1))
            loss2 = 0.25*(loss_function(logits2, Mask[0], logit_distribution2)+loss_function(logits2, Mask[1], logit_distribution2)+loss_function(logits2, Mask[2], logit_distribution2)+loss_function(logits2, Mask[3], logit_distribution2))
            loss = loss1 + loss2
            sum_batch_loss += float(loss)
            loss.backward()
            opt.step()

        train_losses_dict={"loss":sum_batch_loss/(len(train_loader))}
        name1="train"
        #logger.write_to_board(f"{name1}/train_losses", train_losses_dict, epoch+1)
        #ged_dict={"GED":sum_batch_GED/(len(train_loader))}
        #logger.write_to_board(f"{name1}/GED", ged_dict, epoch+1)

        if save_checkpoints == True and (epoch % save_model_step == 0):
            os.makedirs(
                f"saved_models/{meta.directory_name}/{model.name}", exist_ok=True
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "loss": loss,
                },
                f"saved_models/{meta.directory_name}/{model.name}/{epoch+1}_checkpoint.pt",
            )

        # Save the model after last training epoch (for inference or transfer training) to a folder
        """
        Evaluate on the validation set and track to see if overfitting happens
        """
        print("\nValidating")
        sum_Dice = 0
        sum_loss = 0
        counter = 0
        mxD = 0.0
        model.eval()
        with torch.no_grad():
            for images, masks, seg_dist, _ in tqdm(val_loader):
                counter += 1
                # Send tensors to cuda
                images = images.to(device)
                images = images.to(device)
                Mask = []
                Mask1 = []
                Mask2 = []
                images = images.to(device)
                for i in range(4):
                    Mask.append(masks[i].to(device))
                opt.zero_grad()
                # Forward pass
                logit, out = net(images)  # outputs logits
                logits1, logits2, output_dict = model(logit)
                logit_distribution1 = output_dict["distribution1"]
                logit_distribution2 = output_dict["distribution2"]
                # Treshold (default 0.5)

                #Prediction Branch
                for i in range(forward_passes):
                    logits11 = logit_distribution1.sample()
                    sigmoids = torch.sigmoid(logits11)
                    P_mask = (torch.sigmoid(logits11)).ge(meta.masking_threshold)
                    Mask1.append(P_mask)
                for i in range(forward_passes):
                    logits22 = logit_distribution2.sample()
                    sigmoids = torch.sigmoid(logits22)
                    P_mask = (torch.sigmoid(logits22)).ge(meta.masking_threshold)
                    Mask2.append(P_mask)

                #Evidence Branch
                evidence1_1 = torch.sigmoid(logits1)#Foreground Prob.
                evidence1_2 = 1 - evidence1_1#Background Prob.
                #[evidence1, evidence2] = logit map for 2-class softmax
                evidence1_1 = F.softplus(evidence1_1)
                alpha1_1 = evidence1_1 + 1 #Foreground Uncertainty
                evidence1_2 = F.softplus(evidence1_2)
                alpha1_2 = evidence1_2 + 1 #Background Uncertainty

                evidence2_1 = torch.sigmoid(logits2)#Foreground Prob.
                evidence2_2 = 1 - evidence2_1#Background Prob.
                #[evidence1, evidence2] = logit map for 2-class softmax
                evidence2_1 = F.softplus(evidence2_1)
                alpha2_1 = evidence2_1 + 1 #Foreground Uncertainty
                evidence2_2 = F.softplus(evidence2_2)
                alpha2_2 = evidence2_2 + 1 #Background Uncertainty
                # Log images, targets and predictions of the first batch to wandb every 50 epochs
                
                mxD = 0.0
                mxD += compute_max_Dice(Mask2, Mask)
                mxD += compute_max_Dice(Mask1, Mask)
                mxD = 0.5 * mxD
                sum_Dice += mxD
                sum_loss += float(loss)

        val_losses_dict={"loss":sum_loss/(len(val_loader))}
        name1="validation"
        #logger.write_to_board(f"{name1}/validation_losses", val_losses_dict, epoch+1)
        ged_dict={"Dice":sum_Dice/(len(val_loader))}
        #logger.write_to_board(f"{name1}/GED", ged_dict, epoch+1)

        if min_Dice <= sum_Dice:
            min_Dice = sum_Dice
            os.makedirs(f"saved_models/{meta.directory_name}/Attention_Unet+MELT", exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "loss": loss,
                    "epoch": epoch + 1,
                },
                f"saved_models/{meta.directory_name}/Attention_Unet+PLATO/best_model_Dice={min_Dice/(len(val_loader))}.pt",
            )

        print(f"Epoch {epoch+1} Finished")
        print(f"Loss Function:{sum_loss / (len(val_loader))}")
        print(f"Dice:{sum_Dice / (len(val_loader))}\n")
    print(f"Train finished! max_Dice={min_Dice/(len(val_loader))}")

if __name__ == "__main__":
    # Load parsed arguments from command lind
    args = parser.parse_args()

    what_task = args.what
    resume_epoch = args.resume_epoch
    epochs = args.epochs
    batch_size = args.batchsize
    learning_rate = args.lr
    weight_decay = args.weightdecay
    save_checkpoints = args.save_model
    forward_passes = args.N
    log_dir = args.log_dir
    rank = args.rank
    W = bool(args.W)  # Bool for turning off wandb tracking
    transfer_model = args.transfer
    num_filters = args.num_filters
    save_model_step = args.save_model_step
    write = args.write

    # Read in Metadata for the task chosen in command line
    meta_dict = get_meta(what_task)
    meta = SimpleNamespace(**meta_dict)

    # Hand some information about the current run to Wandb Panel
    config = dict(
        epochs=epochs,
        resumed_at=resume_epoch,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        loss="See Paper",
        architecture="SSN",
        dataset=meta.description,
        N_for_metrics=forward_passes,
        rank=rank,
        filter=num_filters,
    )

    training_run_name = (
        str(datetime.now())[:16]
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
    )

    print(f"Modelname: {training_run_name}")
    # Check for GPU

    if torch.cuda.is_available():
        print("\nThe model will be run on GPU.")
    else:
        print("\nNo GPU available!")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"\nUsing the {meta.description} dataset.\n")

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if device == "cuda":
        torch.cuda.manual_seed(230)

    # Init a model
    atten_unet = load_model(model_name='UNet').to(device)
    checkpoint = torch.load("/home/Michael_Bryant/ProbModelBaseline/saved_models/LIDC/Attention_Unet/best_model_Dice=0.6284789358861282.pt")
    atten_unet.load_state_dict(checkpoint["model_state_dict"])

    plato = PLATO(name = training_run_name).to(device)

    opt = optim.AdamW(plato.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Fetch Dataloaders
    train_loader, _ = get_dataloader_2(
        task=what_task, split="train", batch_size=batch_size, shuffle=True, randomsplit=True
    )
    val_loader, _ = get_dataloader_2(
        task=what_task, split="val", batch_size=4, shuffle=False, randomsplit=False
    )
    # Empty GPU Cache
    torch.cuda.empty_cache()
    # Start Training
    train(
        atten_unet,
        plato,
        resume_epoch,
        epochs,
        opt,
        train_loader,
        val_loader,
        save_checkpoints,
        transfer_model,
        meta,
        forward_passes,
        log_dir,
        save_model_step,
        write,
        W=W
    )

    print(f"Saved: {training_run_name} Data: {what_task} Model: SSN")
    # End Training Run

