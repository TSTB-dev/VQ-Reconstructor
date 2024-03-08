import os
from pathlib import Path
from typing import *

import cv2
import torch
import numpy as np
from PIL import Image

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_anomaly_maps(
    anomaly_maps: List[torch.Tensor],
    img_files: List[Path],
    save_dir: str,
    category: str,
    resize_res: Tuple[int, int] = None,
):
    """Save anomaly maps as images.

    Args:
        anomaly_maps (List[torch.Tensor]): Anomaly maps to be saved. Each tensor should has (H, W) shape.
        img_files (List[Path]): Image files corresponding to the anomaly maps. Each image also should has (H, W) shape.
        save_dir_name (str): Directory name to save the anomaly maps.
        category (str): Category name.
        resize_img (Tuple, optional): Resize the anomaly maps to this size. Defaults to None.
    Notes:
        - The saved directory must follow this structure:
            <anomaly_maps_dir>/<object_name>/test/<defect_name>/<image_id>.tiff
        - The saved directory will be created if it does not exist.
        - The saved anomaly maps will be in tiff format.
    """
    
    assert len(anomaly_maps) == len(img_files), "Length of anomaly_maps and img_files must be the same."
    
    # create directory if it does not exist
    save_dir = os.path.join(save_dir, "anomaly_maps")
    if not os.path.exists(save_dir):
        logger.info(f"Directory {save_dir} doesn't exists. Creating...")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
    # resize images if needed
    if resize_res is not None:
        logger.info(f"Resizing anomaly maps to {resize_res}")
        h, w = resize_res
        anomaly_maps = [cv2.resize(am.numpy(), (w, h)) for am in anomaly_maps]
    else:
        anomaly_maps = [am.numpy() for am in anomaly_maps]
    
    test_img = cv2.imread(str(img_files[0]))  
    assert anomaly_maps[0].shape == test_img.shape[:2], f"The resolution of anomaly_maps: {anomaly_maps[0].shape} and img_files: {test_img.shape[:2]} must be the same."
    
    # save anomaly maps to the directory
    for i, anomaly_map in enumerate(anomaly_maps):
        img_file = img_files[i]
        
        logger.debug(img_file.parent.parent.stem)
        save_path = os.path.join(save_dir, category, "test", img_file.parent.name, img_file.stem + ".tiff")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, anomaly_map.astype(np.uint8))
        logger.debug(f"Anomaly map saved to {save_path}")  
    
    logger.info(f"Anomaly maps saved to {save_dir}")
    
if __name__ == "__main__":
    
    dataset_dir = "/home/sakai/projects/VQ-Reconstructor/data/breakfast_box/test"
    category = "breakfast_box"
    save_dir = "./"
    
    img_files = list(Path(dataset_dir).rglob("*.png"))
    test_img = cv2.imread(str(img_files[0]))
    anomaly_maps = [torch.rand(224, 224) for _ in range(len(img_files))]
    
    save_anomaly_maps(
        anomaly_maps=anomaly_maps,
        img_files=img_files,
        save_dir=save_dir,
        category=category,
        resize_res=test_img.shape[:2]
    )
    
    
        
        
    
    