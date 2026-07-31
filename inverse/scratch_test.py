import torch
from models import ConductivityNet, PotentialNet

def main():
    print("Testing ConductivityNet instantiation...")
    cnet = ConductivityNet()
    print("Testing PotentialNet instantiation...")
    pnet = PotentialNet()
    
    print("Loading state_dicts...")
    try:
        cnet.load_state_dict(torch.load("sigma_net.pth", map_location="cpu", weights_only=True))
        print("sigma_net.pth loaded successfully!")
    except Exception as e:
        print(f"Error loading sigma_net.pth: {e}")
        
    try:
        pnet.load_state_dict(torch.load("pot_net.pth", map_location="cpu", weights_only=True))
        print("pot_net.pth loaded successfully!")
    except Exception as e:
        print(f"Error loading pot_net.pth: {e}")

if __name__ == "__main__":
    main()
