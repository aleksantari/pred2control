from models.motor_flow import MotorFlow, FlowTrainConfig, train_motorflow
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


    cfg = FlowTrainConfig(
        L = 150,
        max_steps=2000,
        batch_size=64,
        eval_every=400,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    model = MotorFlow(traj_size=cfg.L).to(device=cfg.device)


    print("cfg.device =", cfg.device)
    print("model device =", next(model.parameters()).device)

    stats = train_motorflow(model, train_ds, test_ds, cfg)

    print("done:", stats)

if __name__ == "__main__":
    main()
    

