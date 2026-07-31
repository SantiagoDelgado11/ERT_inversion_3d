import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

def parse_log_and_plot(log_path, out_path):
    epochs = []
    losses = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        try:
            content = f.read()
        except UnicodeDecodeError:
            with open(log_path, 'r', encoding='utf-16') as f2:
                content = f2.read()
                
    pattern = re.compile(r'loss=([0-9\.e+-]+)')
    
    losses = [float(m.group(1)) for m in pattern.finditer(content)]
    
    # Decimate to avoid huge plots if there are many updates per epoch
    if len(losses) > 1000:
        step = max(1, len(losses) // 500)
        losses = losses[::step]
        
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Total Loss', color='blue', alpha=0.8)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training Loss Curve')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}, total points: {len(losses)}")

if __name__ == "__main__":
    parse_log_and_plot('c:/Users/Doc/Downloads/ERT_inversion_3d/inverse/out.log', 'c:/Users/Doc/Downloads/ERT_inversion_3d/inverse/loss_curve.png')
