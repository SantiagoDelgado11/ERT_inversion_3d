import h5py
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

def main(h5_file):
    with h5py.File(h5_file, 'r') as f:
        # Electrodes
        electrodes = f['electrode_positions'][:]
        
        # Anomaly metadata
        x = f['metadata'].attrs['sphere_x']
        y = f['metadata'].attrs['sphere_y']
        z = f['metadata'].attrs['sphere_z']
        r = f['metadata'].attrs['sphere_r']
        
    fig = go.Figure()
    
    # Add electrodes
    fig.add_trace(go.Scatter3d(
        x=electrodes[:, 0], y=electrodes[:, 1], z=electrodes[:, 2],
        mode='markers',
        marker=dict(size=3, color='black'),
        name='Electrodes'
    ))
    
    # Add anomaly (Sphere)
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    sx = x + r * np.cos(u) * np.sin(v)
    sy = y + r * np.sin(u) * np.sin(v)
    sz = z + r * np.cos(v)
    
    fig.add_trace(go.Surface(
        x=sx, y=sy, z=sz,
        colorscale='Reds',
        showscale=False,
        opacity=0.8,
        name='Anomaly'
    ))
    
    # Update layout
    fig.update_layout(
        title="3D ERT Campaign Visualization",
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Depth (Z) (m)',
            zaxis=dict(autorange='reversed'), # Depth goes down
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    out_html = Path(h5_file).parent / "campaign_3d_interactive.html"
    fig.write_html(str(out_html))
    print(f"Interactive visualization saved to {out_html}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_interactive.py <path_to_campaign.h5>")
    else:
        main(sys.argv[1])
