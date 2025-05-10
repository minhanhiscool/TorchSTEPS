from torchLSTM import ConvLSTM_Encoder_Decoder
import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from misc import random_time
from dataset import RadarDataset
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Generates a random time for the train and validation datasets
    all_times = [random_time() for _ in range(500)]
    train_times, val_times = all_times[:400], all_times[400:]

    # Defines the dataset
    train_dataset = RadarDataset(train_times, T_in, T_out)
    val_dataset = RadarDataset(val_times, T_in, T_out)

    # Loads data for training and validating
    train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=B, shuffle=False)

    # Misc stuff, like defining TensorBoard and save directories
    writer = SummaryWriter(log_dir="../runs")
    model_dir = "../models/"
    os.makedirs(model_dir, exist_ok=True)

    best_val_loss = float("inf")

    # Core of the function
    for epoch in range(num_epoch):
        model.train()  # Sets the model in training mode

        # Defines loss
        train_loss = 0.0
        val_loss = 0.0

        # Iterates over the training data
        for x, m1, m2, m3, ground_truth in train_loader:
            # Prepare data for final time to feed into ConvLSTM_Encoder_Decoder
            x, m1, m2, m3, ground_truth = (
                x.to(device),
                m1.to(device),
                m2.to(device),
                m3.to(device),
                ground_truth.to(device),
            )
            torch.nan_to_num(input=m1, nan=0.0, out=m1)
            torch.nan_to_num(input=m2, nan=0.0, out=m2)
            torch.nan_to_num(input=m3, nan=0.0, out=m3)

            # Forward pass
            out = model(x, m1, m2, m3, ground_truth)

            # Calculate loss
            loss = criterion(out, ground_truth)

            # Backprop the model
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        model.eval()  # Sets the model in evaluation mode
        with torch.no_grad():
            for x, m1, m2, m3, ground_truth in val_loader:
                # Prepare data
                x, m1, m2, m3, ground_truth = (
                    x.to(device),
                    m1.to(device),
                    m2.to(device),
                    m3.to(device),
                    ground_truth.to(device),
                )
                torch.nan_to_num(input=m1, nan=0.0, out=m1)
                torch.nan_to_num(input=m2, nan=0.0, out=m2)
                torch.nan_to_num(input=m3, nan=0.0, out=m3)

                # Forward pass and calculate loss
                out = model(x, m1, m2, m3)
                loss = criterion(out, ground_truth)
                val_loss += loss.item()

        # Get the average loss
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # Log the losses
        writer.add_scalar("Training Loss", train_loss, epoch)
        writer.add_scalar("Validation Loss", val_loss, epoch)

        # Fallback if TensorBoard doesn't work
        print(
            f"Epoch {epoch}/{num_epoch} | Training Loss: {train_loss} | Validation Loss: {val_loss}"
        )

        for param_group in optimizer.param_groups:
            writer.add_scalar("Learning Rate", param_group["lr"], epoch)

        # Found best model? Save it!
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_dir + "best_model.pt")

        scheduler.step(val_loss)


if __name__ == "__main__":
    train(1, 6, 18, 1, 1196, 1219, 500)
