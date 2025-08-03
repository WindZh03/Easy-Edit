import cv2
import torch
import spacy
import re

import numpy as np
from PIL import Image

from huggingface_hub import hf_hub_download
from torchmetrics.multimodal.clip_score import CLIPScore

import warnings
warnings.filterwarnings("ignore")

# Grounding DINO + Segment Anything
from segment_anything import build_sam, SamPredictor
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import load_image, predict


from utils import LLMParse
import utils


# This downloads and builds a GroundingDINO model
def load_model_hf(repo_id, filename, ckpt_config_filename, device='cuda:0'):
    ##! set your own cache_dir path
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename, cache_dir='/home/featurize/work/cache')

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir='/home/featurize/work/cache')
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    _ = model.eval().to(device)
    return model


def join_bboxes(bboxes_xyxy):
    min_x, min_y, _, _ = torch.min(bboxes_xyxy, dim=0)[0]
    _, _, max_x, max_y = torch.max(bboxes_xyxy, dim=0)[0]
    
    return torch.tensor([min_x.item(), min_y.item(), max_x.item(), max_y.item()])


class ExternalMaskExtractor():
    def __init__(self, device, clip_score_threshold=0.) -> None:
        self.device = device
        
        # External Segmentation method = Grounded-SAM (which is GroundingDINO + SAM with bounding box input)
        # First let's load GroundingDINO
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        self.groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename)
        
        # Next, load Segment-Anything
        ##! use your own sam model path
        sam_path = '/home/featurize/work/Grounded-Instruct-Pix2Pix-old/weights/sam_vit_h_4b8939.pth'
        sam = build_sam(checkpoint=sam_path).to(device)
        self.sam_predictor = SamPredictor(sam)
        
        self.llmParse = LLMParse()

            
    def _get_noun_phrases(self, prompt, verbose=False):
        print('edit prompt:',prompt)
        parsed_edit = self.llmParse.get_edit_object(prompt)

        edit_noun_phrases = [str(parsed_edit)]
        print('edit_noun_phrases',edit_noun_phrases)

        if verbose:
            print("Edit Instruction Noun-Phrases:", edit_noun_phrases, '\n')
        return edit_noun_phrases
    
    
    def _dino_predict(self, image, object_name):
        # preprocessing
        transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        image_transformed, _ = transform(image, None)
        boxes, _, _ = predict(model=self.groundingdino_model, image=image_transformed, caption=object_name,
                              box_threshold=0.3, text_threshold=0.25, device='cuda:0')
        
        # account for no bboxes detected case
        return boxes if boxes.shape[0] != 0 else None
    
        

    def genetate_mask(self,image,object_name,promot):
        image_orig = np.asarray(image)
        print('object name',object_name)
        print('image.',type(image))
        boxes = self._dino_predict(image, object_name)   
        
        print('boxes: \n',boxes)
        num_boxes , _ = boxes.shape              # l: number of boxes
        
        if boxes is None:
            return torch.zeros(image_orig.shape[:2], dtype=torch.float32).to('cuda:0')
        
        sorted_boxes = utils.get_order_boxes(boxes)
        print('sorted boxes: \n',sorted_boxes)
        result = self.llmParse.parse_prompt(promot,num_boxes,object_name)
        position,colors,size = utils.ana_prompt(result)
        print(result)
        print('position', position)
        print('color', colors)
        print('size', size)
        boxes_tensor = boxes
        
        if position is not None and colors is None:
            index = int(position-1)
            # sorted_boxes = utils.get_order_boxes(boxes)
            num_boxes , _ = sorted_boxes.shape              # l: number of boxes
            selected_boxes = utils.select_boxes(sorted_boxes,index)
            
            print('boxes: \n',boxes)
            print('sorted_boxes: \n',sorted_boxes)
            print('boxes.number:',num_boxes)
            print('index prompt:',position)
            print('select_boxes: \n',selected_boxes)     
            
            boxes_tensor1 = selected_boxes

        
        if colors is not None:
            color_prompt = colors
            print('color prompt:',colors)
            
            final_boxes = []
            rectangle_img_boxes = utils.get_rectangle_img_boxes(sorted_boxes,image_orig)
            for box, img in rectangle_img_boxes:
                # img 应该是 numpy 数组格式
                img_path = utils.save_numpy_image(img)
                if utils.parse_color(img_path) == color_prompt:
                    final_boxes.append(box)
            
            if position is not None:
                final_tensor2 = []
                n = len(final_boxes)
                print('num of color boxes:',n)
                pos_result = self.llmParse.parse_prompt_position(promot,n,object_name)
                print('pos_result:',pos_result)
                index = int(pos_result)-1
                final_tensor2.append(final_boxes[index])
                boxes_tensor3 = final_tensor2
                boxes_tensor3 = torch.tensor(final_tensor2)
                print('boxes_tensor3:',boxes_tensor3)
            
            if len(final_boxes) == 0:
                boxes_tensor2 = boxes
            else:
                boxes_tensor2 = torch.tensor(final_boxes)  # to device
        
    
        if position is not None and colors is not None:
            boxes_tensor = boxes_tensor3
            print('position is not None and colors is not None')
            print(boxes_tensor)
        elif position is not None:
            boxes_tensor = boxes_tensor1
            print('position is not None')
            print(boxes_tensor)
        elif colors is not None:
            boxes_tensor = boxes_tensor2
            print('colors is not None')
            print(boxes_tensor)
                
        
        self.sam_predictor.set_image(image_orig)
        H, W, _ = image_orig.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_tensor) * torch.Tensor([W, H, W, H])
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_orig.shape[:2]).to('cuda:0')
        masks, _, _ = self.sam_predictor.predict_torch(point_coords=None, point_labels=None,
                                                        boxes=transformed_boxes, multimask_output=False)
        if size is not None:
            n,_ = boxes_tensor.shape
            size = self.llmParse.parse_prompt_size(promot,n,object_name)
            select_index = int(size) - 1
            
            mask_areas = masks.sum(dim=(2, 3)).squeeze(1)
            sorted_indices = torch.argsort(mask_areas, descending=True)
            selected_masks = masks[sorted_indices][select_index]
            
            masks_sum = selected_masks.sum(dim=0).squeeze(0)
            masks_sum[masks_sum > 1] = 1
            
        else:
            masks_sum = sum(masks)[0]
            masks_sum[masks_sum > 1] = 1
            
        return masks_sum
        
        
        
    @torch.no_grad()
    def get_external_mask(self, image, prompt, use_pix=False, mask_dilation_size=11, exclude_noun_phrases=None, verbose=False):
        # Extract all noun-phrases
        prompt_noun_phrases = self._get_noun_phrases(prompt, verbose=verbose)
        chosen_noun_phrase = prompt_noun_phrases[0]
        text_prompt = prompt

        if 'turn' not in prompt or use_pix:
            print('Using Instruct-Pix2Pix')
            print('No noun-phrases detected, falling back to Instruct-Pix2Pix.')
            chosen_noun_phrase = 'IP2P_FALLBACK_MODE'
            width, height = image.size
            external_mask = np.ones((height, width), dtype=np.uint8)
            external_mask = Image.fromarray((255*external_mask).astype(np.uint8))
            return external_mask, prompt

        # Extract its mask 
        print('genetate_mask')  
        print('use my method')
        external_mask = self.genetate_mask(image, chosen_noun_phrase, text_prompt)
        external_mask = cv2.dilate(external_mask.data.cpu().numpy().astype(np.uint8),
                                   kernel=(np.ones((mask_dilation_size, mask_dilation_size), np.uint8)))
        external_mask = Image.fromarray((255*external_mask).astype(np.uint8))
        if verbose:
            print(f'Chose "{chosen_noun_phrase}" as input to G-SAM.')
            external_mask.show()
            
        return external_mask, chosen_noun_phrase