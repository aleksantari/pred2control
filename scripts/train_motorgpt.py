# stage_2/run_train.py
from models.motor_gpt import MotorGPT, TrainConfig, train_motorgpt
import torch



PREPARED_PATH = "data/prepared.pt"  

def main():
    payload = torch.load(PREPARED_PATH, map_location="cpu")

    train_eps_n = payload["train_eps_n"]
    test_eps_n  = payload["test_eps_n"]

    # sanity checks (optional but useful)
    assert len(train_eps_n) > 0 and len(test_eps_n) > 0
    assert train_eps_n[0].shape[1] == 6
    assert train_eps_n[0].dtype in (torch.float32, torch.float64)


    cfg = TrainConfig(
        L=150,
        loss_mode_train="last",
        max_steps= 2000,
        batch_size=64,
        eval_every=250,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    
    model = MotorGPT(traj_size=cfg.L).to(cfg.device)   # should match cfg.L

    print("cfg.L =", cfg.L)
    print("model.traj_size =", model.traj_size)

    print("cfg.device =", cfg.device)
    print("model device =", next(model.parameters()).device)

    stats = train_motorgpt(model, train_eps_n, test_eps_n, cfg)
    print("done:", stats)

if __name__ == "__main__":
    main()
