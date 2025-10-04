import os
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path: str = os.path.join('artifacts', 'preprocessor.pickle')