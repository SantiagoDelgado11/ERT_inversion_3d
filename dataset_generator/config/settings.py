import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class DomainConfig:
    x_length: float = 100.0
    y_length: float = 100.0
    z_length: float = 50.0
    bg_resistivity: float = 100.0

@dataclass
class AnomalyConfig:
    x: float = 50.0
    y: float = 50.0
    z: float = 25.0
    r: float = 10.0
    resistivity: float = 30.0

@dataclass
class SurveyConfig:
    n_electrodes_x: int = 21
    n_electrodes_y: int = 21
    spacing: float = 5.0
    
    # Geometric Constraints for 3D Coverage
    a_skip: int = 2        # Skip factor for A electrode (1=use all, 2=use every other)
    max_n: int = 10        # Max distance factor AM/MN
    max_a_spacing: int = 6 # Max dipole size (MN) multiplier
    
    injected_current: float = 1.0

@dataclass
class MeshConfig:
    core_cell_size: float = 2.0
    padding_cells: int = 10
    padding_factor: float = 1.5

@dataclass
class AppConfig:
    domain: DomainConfig = field(default_factory=DomainConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    survey: SurveyConfig = field(default_factory=SurveyConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    
    output_dir: str = "dataset_output"

def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    if not path.exists():
        return AppConfig()
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
        
    config = AppConfig()
    if 'domain' in data:
        config.domain = DomainConfig(**data['domain'])
    if 'anomaly' in data:
        config.anomaly = AnomalyConfig(**data['anomaly'])
    if 'survey' in data:
        config.survey = SurveyConfig(**data['survey'])
    if 'mesh' in data:
        config.mesh = MeshConfig(**data['mesh'])
        
    config.output_dir = data.get('output_dir', config.output_dir)
    
    return config
