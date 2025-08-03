import random
from http import HTTPStatus
from dashscope import Generation
import dashscope
from urllib.request import urlopen
from alibabacloud_imagerecog20190930.client import Client
from alibabacloud_imagerecog20190930.models import RecognizeImageColorAdvanceRequest
from alibabacloud_tea_openapi.models import Config
from alibabacloud_tea_util.models import RuntimeOptions

import re
import json
import os
import torch
from PIL import Image


##! set your own api key
dashscope.api_key=""
os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID'] = ''
os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET'] = ''

class LLMParse():
    def __init__(self):
        self.model_name = 'qwen-plus'
    
    
    def get_edit_object(self,input_message):
        """function: get the name of the object to be edited
        Args:
            input_message: user input prompt
        Return:
            object name
        """

        Order_Prompt = 'Next, I will enter an image editing statement. Please specify the object to be edited, such as' \
    'For example: please change the grape on the far left into a banana. The object to be edited is the grape, so directly export grape. '\
    'For example: please change the yellow banana into a pencil. The object to be edited is the banana, so directly output banana.Next, I will' \
    'enter a image editing statement. Please follow the example above to output the result directly, just a name'

        messages = [{'role': 'system', 'content': Order_Prompt},
                    {'role': 'user', 'content': input_message},]
        response = Generation.call(model="qwen-plus",
                                messages=messages,
                                seed=1234,
                                temperature=0.8,
                                top_p=0.8,
                                top_k=50,
                                result_format='message')
        if response.status_code == HTTPStatus.OK:
            result = response.output.choices[0].message.content
            # print('edit object: ',result)
            # print(response.message)
            return result
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
    


    def parse_prompt(self,input_message,num,obj):
        
        # v1
        # Order_Prompt = 'Next, I will enter an image editing statement. Please specify the color, relative position, and relative size information of the object to be edited, ' \
        #        'for example: the object to be edited is grape, and there are 5 grapes in total. Please change the red grape in the middle into a banana. From this editing instruction, we can know that the object to be edited is grape, there are 5 grapes in total, its relative position is the middle, which corresponds to the number 3, so output 3, the color is red, so output red, and the relative size is not clearly stated, so output None.' \
        #        'The output format is [(relative position), (color1, color2,...), (relative size)], so directly output [3, red, None].' \
        #        'To give another example, the object to be edited is apple, and there are 7 apples in total. Please change the leftmost largest green and red apples into mangoes, the output should be [1, (green, red), 1], since the relative size is the largest, so output 1, if it is the smallest, it would be 7.' \
        #        'Next, I will enter some image editing statements, please output the color, relative position, and relative size information of the editing object according to the examples and format.'\
        #         'If there is uncertain information, output None directly, do not make assumptions'

        # v2
        Order_Prompt = 'Next, I will enter an image editing statement. Please specify the color, relative position, and relative size information of the object to be edited, ' \
               'for example: the object to be edited is grape, and there are 5 grapes in total. Please change the red grape in the middle into a banana. From this editing instruction, we can know that the object to be edited is grape, there are 5 grapes in total, its relative position is the middle, which corresponds to the number 3, so output 3, the color is red, so output red, and the relative size is not clearly stated, so output None.' \
               'The output format is [relative position, color, relative size], so directly output [3, red, None].' \
               'To give another example, the object to be edited is apple, and there are 7 apples in total. Please change the largest green apple into a mango, the output should be [None, green, 1], since the relative size is the largest, so output 1, if it is the smallest, it would be 7.' \
               'Next, I will enter some image editing statements, please output the color, relative position, and relative size information of the editing object according to the examples and format.'
        

        messages = [{'role': 'system', 'content': Order_Prompt},
                    {'role': 'user', 'content': f'The object to be edited is {obj}, there are {num} in total, {input_message}'},]
        response = Generation.call(model="qwen-plus",
                                messages=messages,
                                # 设置随机数种子seed，如果没有设置，则随机数种子默认为1234
                                #    seed=random.randint(1, 10000),
                                seed=1234,
                                temperature=0.8,
                                top_p=0.8,
                                top_k=50,
                                # 将输出设置为"message"格式
                                result_format='message')
        if response.status_code == HTTPStatus.OK:
            print(response.output.choices[0].message.content)
            result = response.output.choices[0].message.content
            # print(response.message)
            return result
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
            
            
    def parse_prompt_position(self,input_message,num,obj):
        
        # v1
        # Order_Prompt = 'Next, I will enter an image editing statement. Please specify the color, relative position, and relative size information of the object to be edited, ' \
        #        'for example: the object to be edited is grape, and there are 5 grapes in total. Please change the red grape in the middle into a banana. From this editing instruction, we can know that the object to be edited is grape, there are 5 grapes in total, its relative position is the middle, which corresponds to the number 3, so output 3, the color is red, so output red, and the relative size is not clearly stated, so output None.' \
        #        'The output format is [(relative position), (color1, color2,...), (relative size)], so directly output [3, red, None].' \
        #        'To give another example, the object to be edited is apple, and there are 7 apples in total. Please change the leftmost largest green and red apples into mangoes, the output should be [1, (green, red), 1], since the relative size is the largest, so output 1, if it is the smallest, it would be 7.' \
        #        'Next, I will enter some image editing statements, please output the color, relative position, and relative size information of the editing object according to the examples and format.'\
        #         'If there is uncertain information, output None directly, do not make assumptions'

        # v2
        Order_Prompt = 'Next, I will enter an image editing statement. Please specify the relative position of the object to be edited, ' \
               'for example: the object to be edited is grape, and there are 5 grapes in total. Please change red grape in the middle into a banana. From this editing instruction, we can know that the object to be edited is grape, there are 5 grapes in total, its relative position is the middle, which corresponds to the number 3, so output 3 directly.' \
               'So directly output 3.' \
               'To give another example, the object to be edited is apple, and there are 7 apples in total. Please change the third green apple into a mango, the output should be 3.' \
               'Next, I will enter some image editing statements, please output the relative position of the editing object according to the examples and format.'
        

        messages = [{'role': 'system', 'content': Order_Prompt},
                    {'role': 'user', 'content': f'The object to be edited is {obj}, there are {num} in total, {input_message}'},]
        response = Generation.call(model="qwen-plus",
                                messages=messages,
                                # 设置随机数种子seed，如果没有设置，则随机数种子默认为1234
                                #    seed=random.randint(1, 10000),
                                seed=1234,
                                temperature=0.8,
                                top_p=0.8,
                                top_k=50,
                                # 将输出设置为"message"格式
                                result_format='message')
        if response.status_code == HTTPStatus.OK:
            print(response.output.choices[0].message.content)
            result = response.output.choices[0].message.content
            # print(response.message)
            print('color+position:',result)
            return result
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
            
            
    def parse_prompt_size(self,input_message,num,obj):
        
        # v2
        Order_Prompt = 'Next, I will enter an image editing statement. Please specify the relative size of the object to be edited, ' \
               'for example: the object to be edited is grape, and there are 5 grapes in total. Please change the third largest grape in the middle into a banana. From this editing instruction, we can know that the object to be edited is grape, there are 5 grapes in total, its relative size is the third largest, which corresponds to the number 3, so output 3 directly.' \
               'So directly output 3 directly.' \
               'To give another example, the object to be edited is apple, and there are 7 apples in total. Please change the smallest green apple into a mango, the output should be 7.' \
               'Next, I will enter some image editing statements, please output the relative size of the editing object according to the examples and format.'
        

        messages = [{'role': 'system', 'content': Order_Prompt},
                    {'role': 'user', 'content': f'The object to be edited is {obj}, there are {num} in total, {input_message}'},]
        response = Generation.call(model="qwen-plus",
                                messages=messages,
                                # 设置随机数种子seed，如果没有设置，则随机数种子默认为1234
                                #    seed=random.randint(1, 10000),
                                seed=1234,
                                temperature=0.8,
                                top_p=0.8,
                                top_k=50,
                                # 将输出设置为"message"格式
                                result_format='message')
        if response.status_code == HTTPStatus.OK:
            print(response.output.choices[0].message.content)
            result = response.output.choices[0].message.content
            # print(response.message)
            print('color+size:',result)
            return result
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))


def ana_prompt(prompt):
    # 将输入中的 'None' 替换为 'none'，以便后续处理
    prompt = prompt.replace('None', 'none')

    # 使用正则表达式提取元素，支持 'none' 作为有效输入
    match = re.match(r'\[(\s*(\d+|none)\s*),\s*(\s*[^)]+|none)\s*,\s*(\s*(\d+|none)\s*)\]', prompt)

    if match:
        position = match.group(1).strip()
        color = match.group(3).strip() if match.group(3) != 'none' else None  # 只提取一个颜色
        size = match.group(4).strip()

        # 处理 position 和 size
        position = int(position) if position != 'none' else None
        size = int(size) if size != 'none' else None

        # 处理 color
        color = color if color else None  # 如果没有颜色，则返回 [None]
        
        return position, color, size
    
    else:
        print("输入格式不正确")
        return None, None, None



def parse_color(img_path):

    config = Config(
        access_key_id=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_ID'),
        access_key_secret=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
        endpoint='imagerecog.cn-shanghai.aliyuncs.com',
        region_id='cn-shanghai'
    )
    stream = open(img_path, 'rb')
    recognize_image_color_request = RecognizeImageColorAdvanceRequest(
        url_object=stream,
        color_count=3
    )
    runtime = RuntimeOptions()
    try:
        client = Client(config)
        response = client.recognize_image_color_advance(recognize_image_color_request, runtime)
        body = response.body

        result = str(body)
        result = result.replace("'", '"')
        data_dict = json.loads(result)

        color_list = data_dict['Data']['ColorTemplateList']
        max_color = max(color_list, key=lambda x: float(x['Percentage']))
             
        # 输出结果
        print("颜色:", max_color['Label'])
        return max_color['Label']
  
    except Exception as error:
        print(error)
        # print(error.code)
        return None
    
    
def get_order_boxes(boxes):
    sorted_boxes, indices = torch.sort(boxes[:, 0])
    sorted_tensor = boxes[indices]

    return sorted_tensor


def select_boxes(tensor, indices):

    retained_elements = tensor[indices]

    return retained_elements


def get_rectangle_img_boxes(boxes,image):
    height, width, _ = image.shape
    boxes = boxes.numpy()
    rectangle_img_boxes = []
    
    for i, box in enumerate(boxes):
        # if torch.is_tensor(box):
        #     box = box.detach().cpu().numpy()
        x_center, y_center, box_width, box_height = box
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height
        
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)
        
        img_pixels = image[y1:y2, x1:x2]
        rectangle_img_boxes.append((box,img_pixels))
    
    return rectangle_img_boxes



def find_same(tensor1, tensor2, tolerance=1e-6):
    common_rows = []
    
    # 对tensor1的每一行
    for row1 in tensor1:
        # 检查是否在tensor2中存在相同的行
        for row2 in tensor2:
            # 检查整行是否相等（考虑容差）
            if torch.all(torch.abs(row1 - row2) < tolerance):
                common_rows.append(row1)
                break
    
    # 如果找到相同的行，将它们堆叠成二维tensor
    if common_rows:
        return torch.stack(common_rows)
    else:
        # 如果没有相同的行，返回空的二维tensor
        return torch.zeros((0, tensor1.shape[1]))
    
    

def save_image(image_data, file_name='cache', directory='_cache'):

    print('image_data', type(image_data))
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    image_data = Image.fromarray(image_data)
    print('image_data', type(image_data))
    file_path = os.path.join(directory, file_name)
    image_data.save(file_path)

    return file_path


def save_numpy_image(image_array, file_name='cache.png', directory='_cache'):
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    image_data = Image.fromarray(image_array)
    file_path = os.path.join(directory, file_name)
    image_data.save(file_path)

    return file_path



def save_image(image, save_path='saved_images/edited_image.png'):
    """
    将PIL图像保存到指定路径。

    :param image: PIL.Image.Image 对象
    :param save_path: 保存图像的完整路径（包括文件名和扩展名）
    :return: 保存的文件路径
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存图像
    image.save(save_path)
    return save_path