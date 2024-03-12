

import glob
import hydra 
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path='allegro_sim/configs', config_name='preprocess')
def main(cfg : DictConfig) -> None:

    # Initialize the preprocessor module
    prep_module = hydra.utils.instantiate(cfg.preprocessor_module)
    prep_module.apply()
    return

    
if __name__ == '__main__':
    main()