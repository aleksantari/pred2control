from models.motor_flow_chunk import MotorFlow_chunk, FlowTrainConfig, train_motorflow_chunk
import torch


PREPARED_PATH = "data/prepared.pt"

def main():
    payload = torch.load(PREPARED_PATH, map_location="cpu")

    train_ds = payload["train_eps_n"]
    test_ds = payload["test_eps_n"]

     # sanity checks (optional but useful)
    assert len(train_ds) > 0 and len(test_ds) > 0
    assert train_ds[0].shape[1] == 6
    assert train_ds[0].dtype in (torch.float32, torch.float64)


    cfg = FlowTrainConfig()
    device = cfg.device
    model = MotorFlow_chunk().to(device)

    print("model device =", next(model.parameters()).device)

    stats = train_motorflow_chunk(model, train_ds, test_ds, cfg)

    print("done:", stats)

if __name__ == "__main__":
    main()
    

