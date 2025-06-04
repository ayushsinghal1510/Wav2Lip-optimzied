from torch import Tensor 
from numpy import ndarray
import torch
import cv2

def read_img(
    file : str | Tensor | ndarray , 
    rgb : bool = True
) -> ndarray : 
    
        if isinstance(file , str) : return (
            cv2.imread(file) if not rgb 
            else cv2.imread(file)[... , ::-1]
        )

        elif torch.is_tensor(file) : return (
            file.cpu().numpy()[... , ::-1].copy() if not rgb 
            else file.cpu().numpy()
        )

        elif isinstance(file , ndarray) : return (
            file[... , ::-1].copy() if not rgb 
            else file
        )