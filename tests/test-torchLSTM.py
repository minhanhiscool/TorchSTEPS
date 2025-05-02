import torch
from torchLSTM import ConvLSTM_Encoder_Decoder


def test_forward_output_shape():
    B, T_in, T_out, C, H, W = 1, 4, 18, 1, 1219, 1196

    model = ConvLSTM_Encoder_Decoder(
        input_dim=C, hidden_dim=[16, 16], kernel_size=[(3, 3), (3, 3)]
    )
    model.eval()

    x = torch.randn(B, T_in, C, H, W)
    m1 = torch.randn(B, T_out, 1, H, W)
    m2 = torch.randn(B, T_out, 1, H, W)
    m3 = torch.randn(B, T_out, 1, H, W)

    with torch.no_grad():
        out = model(x, m1, m2, m3)

    assert out.shape == (B, T_out, 1, H, W)


def test_non_zero_output():
    B, T_in, T_out, C, H, W = 1, 4, 18, 1, 1219, 1196
    model = ConvLSTM_Encoder_Decoder(
        input_dim=C, hidden_dim=[16, 16], kernel_size=[(3, 3), (3, 3)]
    )
    model.eval()

    x = torch.randn(B, T_in, C, H, W)
    m1 = torch.randn(B, T_out, 1, H, W)
    m2 = torch.randn(B, T_out, 1, H, W)
    m3 = torch.randn(B, T_out, 1, H, W)

    with torch.no_grad():
        out = model(x, m1, m2, m3)

    assert out.sum() != 0, "Output is all zeros, which is unexpected"


def test_teacher_forcing_behavior():
    B, T_in, T_out, C, H, W = 1, 4, 18, 1, 1219, 1196
    model = ConvLSTM_Encoder_Decoder(
        input_dim=C, hidden_dim=[16, 16], kernel_size=[(3, 3), (3, 3)]
    )
    model.eval()

    x = torch.randn(B, T_in, C, H, W)
    m1 = torch.randn(B, T_out, 1, H, W)
    m2 = torch.randn(B, T_out, 1, H, W)
    m3 = torch.randn(B, T_out, 1, H, W)
    ground_truth = torch.randn(B, T_out, 1, H, W)

    # With teacher forcing (ground_truth is provided)
    out_with_teacher_forcing = model(x, m1, m2, m3, ground_truth=ground_truth)

    # Without teacher forcing (no ground_truth)
    model.teacher_forcing_ratio = 0  # Turn off teacher forcing
    out_without_teacher_forcing = model(x, m1, m2, m3)

    assert not torch.allclose(out_with_teacher_forcing, out_without_teacher_forcing), (
        "Teacher forcing didn't affect the output as expected"
    )


def test_variable_input_size():
    for H, W in [(1219, 1196), (256, 256), (512, 512)]:
        B, T_in, T_out, C = 1, 4, 18, 1
        model = ConvLSTM_Encoder_Decoder(
            input_dim=C, hidden_dim=[16, 16], kernel_size=[(3, 3), (3, 3)]
        )
        model.eval()

        x = torch.randn(B, T_in, C, H, W)
        m1 = torch.randn(B, T_out, 1, H, W)
        m2 = torch.randn(B, T_out, 1, H, W)
        m3 = torch.randn(B, T_out, 1, H, W)

        with torch.no_grad():
            out = model(x, m1, m2, m3)

        assert out.shape == (B, T_out, 1, H, W), (
            f"Failed for input size {H}x{W}. Expected shape (B, T_out, 1, H, W), got {out.shape}"
        )


def test_gradients_propagation():
    B, T_in, T_out, C, H, W = 1, 4, 18, 1, 1219, 1196
    model = ConvLSTM_Encoder_Decoder(
        input_dim=C, hidden_dim=[16, 16], kernel_size=[(3, 3), (3, 3)]
    )
    model.train()  # Ensure we are in training mode

    x = torch.randn(B, T_in, C, H, W, requires_grad=True)
    m1 = torch.randn(B, T_out, 1, H, W)
    m2 = torch.randn(B, T_out, 1, H, W)
    m3 = torch.randn(B, T_out, 1, H, W)

    out = model(x, m1, m2, m3)

    # Check gradients
    out.sum().backward()
    assert x.grad is not None, "Gradients not propagating"


def test_memory_usage():
    try:
        B, T_in, T_out, C, H, W = 1, 4, 18, 1, 1219, 1196
        model = ConvLSTM_Encoder_Decoder(
            input_dim=C, hidden_dim=[16, 16], kernel_size=[(3, 3), (3, 3)]
        )
        model.eval()

        x = torch.randn(B, T_in, C, H, W)
        m1 = torch.randn(B, T_out, 1, H, W)
        m2 = torch.randn(B, T_out, 1, H, W)
        m3 = torch.randn(B, T_out, 1, H, W)

        with torch.no_grad():
            out = model(x, m1, m2, m3)

        print(f"Output shape: {out.shape}")
        # You can use the following to check the memory usage
        # torch.cuda.memory_allocated() if using GPU

    except RuntimeError as e:
        print(f"Memory error encountered: {e}")
        assert False, "Test failed due to memory error"


def test_edge_case_with_zeros():
    B, T_in, T_out, C, H, W = 1, 4, 18, 1, 1219, 1196
    model = ConvLSTM_Encoder_Decoder(
        input_dim=C, hidden_dim=[16, 16], kernel_size=[(3, 3), (3, 3)]
    )
    model.eval()

    x = torch.zeros(B, T_in, C, H, W)
    m1 = torch.zeros(B, T_out, 1, H, W)
    m2 = torch.zeros(B, T_out, 1, H, W)
    m3 = torch.zeros(B, T_out, 1, H, W)

    with torch.no_grad():
        out = model(x, m1, m2, m3)

    assert out.shape == (B, T_out, 1, H, W), (
        f"Expected shape (B, T_out, 1, H, W), but got {out.shape}"
    )
    assert torch.all(out == 0), "Expected output to be zero when input is all zeros"
