import litserve as ls
# from torch.utils.model_zoo import load_url
from torch.hub import load_state_dict_from_url

from scripts.models.models import s3fd
from scripts.services.services import read_img
from numpy import ndarray
import numpy as np 
import torch 
from torch import Tensor
import torch.nn.functional as F

def decode(
    loc : Tensor , 
    priors : Tensor , 
    variances : list = [0.1 , 0.2]
) -> Tensor : 

    boxes : Tensor = torch.cat(
        (
            priors[: , : 2] + 
            loc[: , :2] * 
            variances[0] * 
            priors[: , 2:] , 
            
            priors[: , 2 :] * 
            torch.exp(loc[: , 2 :] * variances[1])
        ) , 1)

    boxes[: , : 2] -= boxes[: , 2 :] / 2
    boxes[: , 2 :] += boxes[: , : 2]

    return boxes

def preprocess_image(raw_img : ndarray) -> Tensor : 

    normalized_img : ndarray = raw_img - np.array([104 , 117 , 123])
    redimensioned_img : ndarray = normalized_img.transpose(2 , 0 , 1)
    batched_img : ndarray = redimensioned_img.reshape((1 ,) + redimensioned_img.shape)

    img_tensor : Tensor = torch.from_numpy(batched_img).float()

    return img_tensor

class InferencePipeline(ls.LitAPI) : 

    def setup(self , device) -> None :
        
        model_weights : dict = load_state_dict_from_url('https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth')

        name_mapping : dict = {
            # Conv1 block
            'conv1_1.weight' : 'conv1.conv1.weight' ,
            'conv1_1.bias' : 'conv1.conv1.bias' ,
            'conv1_2.weight' : 'conv1.conv2.weight' ,
            'conv1_2.bias' : 'conv1.conv2.bias' ,
            
            # Conv2 block
            'conv2_1.weight' : 'conv2.conv1.weight' ,
            'conv2_1.bias' : 'conv2.conv1.bias' ,
            'conv2_2.weight' : 'conv2.conv2.weight' ,
            'conv2_2.bias' : 'conv2.conv2.bias' ,
            
            # Conv3 block
            'conv3_1.weight' : 'conv3.conv1.weight' ,
            'conv3_1.bias' : 'conv3.conv1.bias' ,
            'conv3_2.weight' : 'conv3.conv2.weight' ,
            'conv3_2.bias' : 'conv3.conv2.bias' ,
            'conv3_3.weight' : 'conv3.conv3.weight' ,
            'conv3_3.bias' : 'conv3.conv3.bias' ,
            
            # Conv4 block
            'conv4_1.weight' : 'conv4.conv1.weight' ,
            'conv4_1.bias' : 'conv4.conv1.bias' ,
            'conv4_2.weight' : 'conv4.conv2.weight' ,
            'conv4_2.bias' : 'conv4.conv2.bias' ,
            'conv4_3.weight' : 'conv4.conv3.weight' ,
            'conv4_3.bias' : 'conv4.conv3.bias' ,
            
            # Conv5 block
            'conv5_1.weight' : 'conv5.conv1.weight' ,
            'conv5_1.bias' : 'conv5.conv1.bias' ,
            'conv5_2.weight' : 'conv5.conv2.weight' ,
            'conv5_2.bias' : 'conv5.conv2.bias' ,
            'conv5_3.weight' : 'conv5.conv3.weight' ,
            'conv5_3.bias' : 'conv5.conv3.bias' ,
            
            # Conv6 block 
            'fc6.weight' : 'conv6.conv1.weight' ,
            'fc6.bias' : 'conv6.conv1.bias' ,
            'fc7.weight' : 'conv6.conv2.weight' ,
            'fc7.bias' : 'conv6.conv2.bias' ,
            
            # Conv7 block
            'conv6_1.weight' : 'conv7.conv1.weight' ,
            'conv6_1.bias' : 'conv7.conv1.bias' ,
            'conv6_2.weight' : 'conv7.conv2.weight' ,
            'conv6_2.bias' : 'conv7.conv2.bias' ,
            
            # Conv8 block
            'conv7_1.weight' : 'conv8.conv1.weight' ,
            'conv7_1.bias' : 'conv8.conv1.bias' ,
            'conv7_2.weight' : 'conv8.conv2.weight' ,
            'conv7_2.bias' : 'conv8.conv2.bias' ,
        }
        
        new_state_dict : dict = {}
        
        for old_name , new_name in name_mapping.items() : 

            if old_name in model_weights : new_state_dict[new_name] = model_weights[old_name]
        for key , value in model_weights.items() : 

            if key not in name_mapping and key not in new_state_dict : new_state_dict[key] = value

        self.model : s3fd = s3fd()

        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        
    # def decode_request(self , request : dict ) -> str : 

    #     return request['file_path']

    def predict(self , request : dict) : 

        file_path : str = request['file_path']
        raw_img : ndarray = read_img(file_path)

        img_tensor : Tensor = preprocess_image(raw_img)

        with torch.no_grad() :

            for index , (cls_ , reg) in enumerate(self.model(img_tensor)) :

                cls_ : Tensor = cls_.detach().cpu()
                reg : Tensor = reg.detach().cpu()

                softmax_reg : Tensor = F.softmax(reg , dim = 1)

                stride : int = 2 ** (index + 2)

                position : zip = zip(*np.where(cls_[: , 1 , : , :] > 0.05))

                for _ , height_index , width_index in position :

                    axc , ayc = stride / 2 + width_index * stride , stride / 2 + height_index * stride
                    score : Tensor = cls_[0 , 1 , height_index , width_index]
                    raw_score : float = score.numpy().tolist()

                    loc : Tensor = softmax_reg[0 , : , height_index , width_index].contiguous().view(1 , 4)
                    priors : Tensor = torch.Tensor([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])

                    box : Tensor = decode(loc , priors)
                    raw_box : list = box.numpy().tolist()

                    x1 , y1 , x2 , y2 = raw_box[0]

                    yield (index , x1 , y1 , x2 , y2 , raw_score)

    def encode_response(self , output) : 

        for out in output : yield {'output' : out}

if __name__ == '__main__' : 
    
    server = ls.LitServer(
        InferencePipeline(max_batch_size = 1) , 
        accelerator = 'auto' , 
        stream = True
    )
    
    server.run(port = 8000)