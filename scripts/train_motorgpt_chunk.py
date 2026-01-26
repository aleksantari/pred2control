# stage_2/run_train.py
from models.motor_gpt_chunk import Motorgpt_chunk, TrainConfig, train_motorgpt_chunk
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


    cfg = TrainConfig()
    device = cfg.device
    
    model = Motorgpt_chunk().to(device)  

  
    print("model device =", next(model.parameters()).device)

    stats = train_motorgpt_chunk(model, train_eps_n, test_eps_n, cfg=cfg)
    print("done:", stats)

if __name__ == "__main__":
    main()
