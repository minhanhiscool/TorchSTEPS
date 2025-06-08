from torchLSTM import ConvLSTM_Encoder_Decoder
import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from misc import resize_batch, all_time
from dataset import RadarDataset
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def custom_mse(pred, target, threshold=0.0033, alpha=0.3, beta=0.7, mode="linear"):
    """
    A custom loss function to compute the loss between the predictions and the ground truth

    Args:
        pred (Tensor): Predictions from the model
        target (Tensor): Ground truth from the dataset
        threshold (float, optional): Threshold for the loss function. Default is 0.0033
        alpha (float, optional): Weight for the MSE loss. Default is 0.3
        beta (float, optional): Weight for the threshold loss. Default is 0.7
        mode (str, optional): Mode for the threshold loss. Accepts "linear" and "exponential". Default is "linear"

    Returns:
        Tensor: The loss between the predictions and the ground truth
    """

    B, T, C, H, W = pred.shape
    if mode == "linear":
        weights = torch.linspace(10.0, 1.0, T).view(1, T, 1, 1, 1).to(device)
    elif mode == "exponential":
        weights = (
            torch.tensor([(1.2**i) for i in range(T)]).view(1, T, 1, 1, 1).to(device)
        )
    else:
        raise ValueError("Mode must be either 'linear' or 'exponential'")

    def threshold_loss(pred, target, threshold):
        mask = target > threshold
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        return (((pred - target) ** 2) * weights)[mask].mean()

    mse = (((pred - target) ** 2) * weights).mean()
    threshold_mse = threshold_loss(pred, target, threshold)
    return alpha * mse + beta * threshold_mse


def csi_score(pred, target, threshold=0.0033):
    """
    The CSI score for a given prediction and target
    Args:
        pred (Tensor): Predictions from the model
        target (Tensor): Ground truth from the dataset
        threshold (float, optional): Threshold for the loss function. Default is 0.0033
    """
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()

    TP = pred_bin * target_bin
    FP = pred_bin * (1 - target_bin)
    FN = (1 - pred_bin) * target_bin

    TP_sum = TP.sum(dim=[0, 2, 3, 4])
    FP_sum = FP.sum(dim=[0, 2, 3, 4])
    FN_sum = FN.sum(dim=[0, 2, 3, 4])

    csi = TP_sum / (TP_sum + FP_sum + FN_sum + 1e-6)
    return csi.mean()


def log_video(pred, ground_truth, writer):
    """
    Logging function to log the predictions and ground truth videos to tensorboard

    Args:
        pred (Tensor): Predictions from the model
        ground_truth (Tensor): Ground truth from the dataset
    """
    B, T, C, H, W = pred.shape
    from torchvision.utils import make_grid

    pred_grid = make_grid(pred.reshape(B * T, C, H, W).clamp(0, 1), nrow=T)
    gt_grid = make_grid(ground_truth.reshape(B * T, C, H, W).clamp(0, 1), nrow=T)
    writer.add_image("Val/Grid_Predictions", pred_grid, global_step=0)
    writer.add_image("Val/Grid_GroundTruth", gt_grid, global_step=0)


def train(B, T_in, T_out, C, H, W, num_epoch=1000):
    """
    Training Loop for torchLSTM
    The training loop is a for loop that iterates over the number of epochs given
    It loads using dataset.py the train and validation datasets (see dataset.py for further info)
    Then trains the model using the dataset given

    Args:
        B (int): Batch size
        T_in (int): Input sequence length
        T_out (int): Output sequence length
        C (int): Input channel
        H (int): Height of the input
        W (int): Width of the input
        num_epoch (int, optional): Number of epochs. Defaults to 1000.
    """

    # The model in question
    model = ConvLSTM_Encoder_Decoder(
        input_dim=C, hidden_dim=[16, 16], kernel_size=[(3, 3), (3, 3)]
    )
    model = model.to(device)

    # Defines all of the juicy stuff for training
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Generates a random time for the train and validation datasets
    all_times = all_time()
    bound = int(len(all_times) * 0.8)

    train_dataset = RadarDataset(all_times[:bound], T_in, T_out, H, W)
    train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True)

    val_dataset = RadarDataset(all_times[bound:], T_in, T_out, H, W)

    val_loader = DataLoader(val_dataset, batch_size=B, shuffle=True)

    # Misc stuff, like defining TensorBoard and save directories
    writer = SummaryWriter(log_dir="../runs")
    model_dir = "../models/"
    os.makedirs(model_dir, exist_ok=True)

    best_val_loss = float("inf")

    # Core of the function
    for epoch in range(num_epoch):
        # Defines loss
        train_loss = 0.0

        model.train()  # Sets the model in training mode

        os.system(
            f"tmux set -g status-right 'Epoch {epoch + 1}/{num_epoch} | "
            f"Batch {1}/{len(train_loader)} | Training'"
        )

        # Iterates over the training data
        for idx, (x, m1, m2, m3, ground_truth) in enumerate(train_loader):
            # Prepare data for final time to feed into ConvLSTM_Encoder_Decoder
            x, m1, m2, m3, ground_truth = (
                x.to(device),
                m1.to(device),
                m2.to(device),
                m3.to(device),
                ground_truth.to(device),
            )
            # I hate resize_batch
            x = resize_batch(x)
            m1 = resize_batch(m1)
            m2 = resize_batch(m2)
            m3 = resize_batch(m3)
            ground_truth = resize_batch(ground_truth)

            print("All tensors are ready!")

            # Forward pass
            out = model(x, m1, m2, m3, ground_truth=ground_truth)

            # Calculate loss
            loss = custom_mse(out, ground_truth)

            log_video(out.detach(), ground_truth.detach(), writer)

            print(
                f"Epoch {epoch + 1}/{num_epoch} | Batch {idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}"
            )

            # Backprop the model
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

            os.system(
                f"tmux set -g status-right 'Epoch {epoch + 1}/{num_epoch} | "
                f"Batch {idx + 2}/{len(train_loader)} | Training'"
            )
        model.eval()  # Sets the model in evaluation mode

        val_loss_mse = 0.0
        val_loss_csi = 0.0

        os.system(
            f"tmux set -g status-right 'Epoch {epoch + 1}/{num_epoch} | "
            f"Batch {1}/{len(val_loader)} | Evaluating'"
        )

        with torch.no_grad():
            for idx, (x, m1, m2, m3, ground_truth) in enumerate(val_loader):
                # Prepare data
                x, m1, m2, m3, ground_truth = (
                    x.to(device),
                    m1.to(device),
                    m2.to(device),
                    m3.to(device),
                    ground_truth.to(device),
                )

                # I still hate resize_batch
                x = resize_batch(x)
                m1 = resize_batch(m1)
                m2 = resize_batch(m2)
                m3 = resize_batch(m3)
                ground_truth = resize_batch(ground_truth)

                print("All tensors are ready!")

                # Forward pass and calculate loss
                out = model(x, m1, m2, m3)
                loss_mse = custom_mse(out, ground_truth)
                loss_csi = csi_score(out, ground_truth)
                val_loss_mse += loss_mse.item()
                val_loss_csi += loss_csi.item()

                log_video(out, ground_truth, writer)

                print(
                    f"Epoch {epoch + 1}/{num_epoch} | Batch {idx + 1}/{len(val_loader)} | Loss: {loss_mse.item():.4f}"
                )

                os.system(
                    f"tmux set -g status-right 'Epoch {epoch + 1}/{num_epoch} | "
                    f"Batch {idx + 2}/{len(val_loader)} | Evaluating'"
                )

        # Get the average loss
        train_loss /= len(train_loader)
        val_loss_mse /= len(val_loader)
        val_loss_csi /= len(val_loader)

        # Log the losses
        writer.add_scalar("Training Loss", train_loss, epoch)
        writer.add_scalar("Validation Loss - MSE", val_loss_mse, epoch)
        writer.add_scalar("Validation Loss - CSI", val_loss_csi, epoch)

        # Fallback if TensorBoard doesn't work
        print(
            f"Epoch {epoch}/{num_epoch} | Training Loss: {train_loss} | Validation Loss - MSE: {val_loss_mse} | Validation Loss - CSI: {val_loss_csi}"
        )
        for i in scheduler.get_last_lr():
            writer.add_scalar("Learning Rate", i, epoch)

        # Found best model? Save it!
        if val_loss_mse < best_val_loss:
            best_val_loss = val_loss_mse
            torch.save(model.state_dict(), model_dir + "best_model.pt")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"{model_dir}/checkpoint_epoch{epoch}.pt")

        scheduler.step(val_loss_mse)


if __name__ == "__main__":
    train(1, 6, 18, 1, 1196, 1219, 500)
