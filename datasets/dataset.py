import json
import math
import os
import random
from collections import deque
from typing import List, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import tifffile as tiff
import torch
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2
from PIL import Image, UnidentifiedImageError

from .cutpaste import CutPaste, PerlinPaste


def get_transform(args, mean = (0,0,0), std = (1,1,1)):
    train_transform = A.Compose([
                A.ToRGB(),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5), 
                A.RandomRotate90(p=0.5),
                A.ChannelShuffle(p=0.25),
                A.ToGray(p=0.25),
                A.Resize(width=args.image_size, height=args.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ])
    
    test_transform = A.Compose([
        A.ToRGB(),
        A.Resize(width=args.image_size, height=args.image_size),
    ])
    
    norm_transform = A.Compose([
        A.Normalize(mean = mean, std = std),
        ToTensorV2(),  
    ])

    '''
    target_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor()
    ])
    '''
    #preprocess.transforms[0] = transforms.Resize(size=(args.image_size, args.image_size), 
    #                                             interpolation=transforms.InterpolationMode.BICUBIC,
    #                                            max_size=None, antialias=None)
    #preprocess.transforms[1] = transforms.CenterCrop(size=(args.image_size, args.image_size))

    return train_transform, test_transform, norm_transform




def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    if dataset_name == 'mvtec':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    elif dataset_name == 'MVTec3D':
        obj_list = ['bagel', 'carrot',  'dowel',  'potato', 'rope', 'cable_gland',  'cookie',  
                    'foam', 'peach', 'tire']
    elif dataset_name == 'visa':
        obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif dataset_name == 'MPDD':
        obj_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
    elif dataset_name == 'BTAD':
        obj_list = ['01', '02', '03']
    elif dataset_name == 'DAGM_KaggleUpload':
        obj_list = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']
    elif dataset_name == 'SDD':
        obj_list = ['electrical_commutators']
    elif dataset_name == 'KSDD2':
        obj_list = ['electrical_commutators']
    elif dataset_name == 'RSDD':
        obj_list = ['rsdd']    
    elif dataset_name == 'DTD-Synthetic':
        obj_list = ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125', 'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069']
    elif dataset_name == 'xmz_cropped' or dataset_name == 'xmz':
        obj_list = ['cls1', 'cls2']
    
    elif dataset_name == 'medical-cls':
        obj_list = ['Br35H', 'BrainMRI', 'HeadCT'] #, 'COVID19']
    elif dataset_name == 'medical-seg':
        obj_list = ['ClinicDB',  'ColonDB', 'Endo',  'ISIC', 'Kvasir'] #,  'TN3K']
    elif dataset_name == 'bmad-medical-seg':
        obj_list = ['brain',  'liver',  'retinal']
    elif dataset_name == 'RealIAD':
        obj_list = ["audiojack", "bottle_cap", "button_battery", "end_cap", "eraser", "fire_hood", "mint", "mounts", "pcb", "phone_battery",  \
                    "plastic_nut", "plastic_plug", "porcelain_doll", "regulator", "rolled_strip_base", "sim_card_set", "switch", "tape", "terminalblock",  \
                    "toothbrush", "toy", "toy_brick", "transistor1", "u_block", "usb", "usb_adaptor", "vcpill", "wooden_beads", "woodstick", "zipper"]
    elif dataset_name == 'Real-IAD-Variety':
        obj_list = ["2pin_block_plug", "3_adapter", "3pin_aviation_connector", "4_wire_stepping_motor", "D-sub-connector", "DVD_switch", "LED_indicator", \
                    "PLCC_socket", "VR_joystick", "access_card", "accurate_detection_switch", "aircraft_model_head", "angled_toggle_switch", "audio_jack_socket", \
                    "bag_buckle", "ball_pin", "balun_transformer", "battery", "battery_holder_connector", "battery_socket_connector", "bend_connector", "blade_switch", \
                    "blue_light_switch", "bluetooth_module", "boost_converter_module", "bread_model", "brooch_clasp_accessory", "button_battery_holder", "button_motor", \
                    "button_switch", "car_door_lock_switch", "ceramic_fuse", "ceramic_wave_filter", "charging_port", "chip-inductor", "circuit_breaker", \
                    "circular_aviation_connector", "common-mode-choke", "common-mode-filter", "connector", "connector_housing-female", "console_switch", \
                    "crimp_st_cable_mount_box", "dc_jack", "dc_power_connector", "detection_switch", "duckbill_circuit_breaker", "earphone_audio_unit", \
                    "effect_transistor", "electronic_watch_movement", "ethernet-connector", "ferrite_bead", "ffc_connector_plug", "flow_control_valve", \
                    "flower_copper_shape", "flower_velvet_fabric", "fork-crimp-terminal", "fuse_cover", "fuse_holder", "gear", "gear_motor", "green-ring-filter", \
                    "hairdryer_switch", "hall_effect_sensor", "headphone_jack-female", "headphone_jack_socket", "hex_plug", "humidity_sensor", "ingot_buckle", \
                    "insect_metal_parts", "inverter_connector", "jam_jar_model", "joystick_switch", "kfc_push-key_switch", "knob-cap", "laser_diode", "lattice_block_plug", \
                    "lego-pin-connector-plate", "lego-propeller", "lego-reel", "lego-technical-gear", "lego-turbine", "lighting_connector", "lilypad_led", "limit-switch", \
                    "lithium_battery_plug", "littel-fuse", "little_cow_model", "lock", "long-zipper", "meteor_hammer_arrowhead", "miniature_laser_module", \
                    "miniature_lifting_motor", "miniature_motor", "miniature_stepper_motor", "mobile_charging_connector", "model_steering_module", "monitor_socket", \
                    "motor_bracket", "motor_gear_reducer", "motor_plug", "mouse_socket", "multi_function_switch", "nylon_ball_head", "optical_fiber_outlet", \
                    "pencil_sharpener", "pinboard_connector", "pitch_connector", "pneumatic_elbow", "pot_core", "potentiometer", "power_bank_module", \
                    "power_inductor", "power_jack", "power_strip_socket", "pulse_transformer", "purple-clay-pot", "push_button_switch", "push_in_terminal", \
                    "recorder_switch", "rectangular-connector-accessories", "red_terminal", "retaining_ring", "rheostat", "rotary_position_sensor", \
                    "round_twist_switch", "self-lock_switch", "shelf_support_module", "side_press_switch", "side_release_buckle", "silicon_cell_sensor", "sim_card_reader", \
                    "single-pole_potentiometer", "single_switch", "slide_switch", "slipper- model", "small_leaf", "smd_receiver_module", "solderless_adapter", \
                    "spherical_airstone", "spring_antenna", "square_terminal", "steering_T-head", "steering_wheel", "suction-cup", "telephone_spring_switch", \
                    "tension_snap_release_clip", "thumbtack", "thyristor", "toy-tire", "traceless_hair_clip", "travel_switch", "travel_switch_green", \
                    "tubular_key_switch", "vacuum_switch", "vehicle_harness_conductor", "vertical-adjustable-resistor", "vibration_motor", \
                    "volume_potentiometer", "wireless_receiver_module"]
    else:
        raise NotImplementedError('Dataset not implemented: {}'.format(dataset_name))
         

    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id

# Mosaic Augmentation
# https://github.com/snuviplab/NeMo/blob/main/dataloader.py
def mosaic_aug(imgs_list, masks_list, img_size=512):
    num_bgs = len(imgs_list)
    denum = int(math.sqrt(num_bgs))

    if  np.random.uniform() > 0.5:
        quarter_sz = img_size//denum
        crs_y = np.random.randint(quarter_sz, img_size+1-quarter_sz)
        crs_x = np.random.randint(quarter_sz, img_size+1-quarter_sz)
    else:
        crs_y = img_size//denum
        crs_x = img_size//denum

    imgs_resized = []
    masks_resized = []
    for i in range(num_bgs):
        row = i // denum
        col = i % denum
        y_start = row * crs_y
        y_end = y_start + crs_y
        x_start = col * crs_x
        x_end = x_start + crs_x
        
        if y_end > img_size or x_end > img_size:
            if y_end > img_size:
                crs_y = y_end - y_start
            if x_end > img_size:
                crs_x = x_end - x_start
                
        resize_transform = A.Compose([
                A.Resize(crs_y, crs_x, always_apply=True)
            ])
    
        temp = resize_transform(image=np.array(imgs_list[i]), mask=np.array(masks_list[i]))
        img_resized = temp['image']
        mask_resized = temp['mask']

        imgs_resized.append(img_resized)
        masks_resized.append(mask_resized)

    imgs = imgs_resized
    masks = masks_resized
    
    num_rows = num_cols = int(math.sqrt(num_bgs))
    idxs = [(i*num_cols,i*num_cols+num_cols) for i in range(num_rows)]
    img = [np.concatenate(imgs[_from:_to], axis=1) for (_from, _to) in idxs]
    img = np.concatenate(img, axis=0).astype(np.uint8)

    mask = [np.concatenate(masks[_from:_to], axis=1) for (_from, _to) in idxs]
    mask = np.concatenate(mask, axis=0).astype(np.uint8)

    img = Image.fromarray(img.astype(np.uint8))
    mask = Image.fromarray(mask.astype(np.uint8))

    return img, mask



'''
def random_crop_with_mask_pil(img: Image.Image, mask) -> Image.Image:
    """
    对异常图像，以异常mask中任意一个连通区域的中心进行裁剪，裁剪尺寸在 [1/3, 1/2]，并带有中心扰动。
    - img: PIL.Image
    - mask: PIL.Image (模式 'L') 或 numpy.ndarray (H, W)，非零表示异常区域
    - 返回裁剪后的 PIL.Image
    """
    W, H = img.size
    short_side = min(W, H)

    # 将 mask 转换为 2D 的 numpy 数组
    if isinstance(mask, Image.Image):
        mask_np = np.array(mask)
    elif isinstance(mask, np.ndarray):
        mask_np = mask
    else:
        raise TypeError("mask must be a PIL.Image or numpy.ndarray")
    
    if mask_np.ndim == 3:
        # 如为多通道，转为灰度
        mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
    # 使用 OpenCV 8 连通组件标题提取
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    # label 0 是背景，若只有背景则退回普通裁剪
    crop_min = max(1, (3 * short_side) // 4)
    crop_max = max(1, short_side)
    crop_size = random.randint(crop_min, crop_max)
    if num_labels <= 1:
        x = random.randint(0, max(0, W - crop_size))
        y = random.randint(0, max(0, H - crop_size))

        return img.crop((x, y, x + crop_size, y + crop_size)), mask.crop((x, y, x + crop_size, y + crop_size))
    else:
        # 随机选择一个非背景连通区域（1..num_labels-1）
        chosen_label = random.randint(1, num_labels - 1)

        # 该区域的质心作为裁剪中心
        cx, cy = centroids[chosen_label]
        cx = int(float(cx))
        cy = int(float(cy))

        # 在异常中心周围添加扰动
        dx = int(np.random.uniform(-crop_size / 4, crop_size / 4))
        dy = int(np.random.uniform(-crop_size / 4, crop_size / 4))

        x = int(cx - crop_size / 2) + dx
        y = int(cy - crop_size / 2) + dy

        # 保证裁剪区域在图像内部
        x = max(0, min(x, W - crop_size))
        y = max(0, min(y, H - crop_size))

        return img.crop((x, y, x + crop_size, y + crop_size)), mask.crop((x, y, x + crop_size, y + crop_size))
'''


def random_crop_with_mask_pil(img: Image.Image, mask) -> (Image.Image, Image.Image):
    """
    以异常mask中任意一个连通区域的矩形框，向外以该矩形框最长边的2~3倍随机延伸裁剪，
    并裁剪img和mask。

    - img: PIL.Image
    - mask: PIL.Image（模式 'L'） 或 numpy.ndarray (H, W)，非零表示异常区域
    - 返回裁剪后的 (img_crop, mask_crop)
    """
    W, H = img.size
    min_crop_size = 256

    # 将 mask 转换为 2D 的 numpy 数组
    if isinstance(mask, Image.Image):
        mask_np = np.array(mask)
    elif isinstance(mask, np.ndarray):
        mask_np = mask
    else:
        raise TypeError("mask must be a PIL.Image or numpy.ndarray")

    if mask_np.ndim == 3:
        # 如为多通道，转为灰度
        mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)

    # 使用 OpenCV 8 连通组件提取
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    # label 0 是背景，若只有背景则退回普通裁剪
    if num_labels <= 1:
        # 无异常区域，随机裁剪整个图像
        '''
        short_side = min(W, H)
        crop_min = max(1, (3 * short_side) // 4)
        crop_max = max(1, short_side)
        crop_size = random.randint(crop_min, crop_max)
        x = random.randint(0, max(0, W - crop_size))
        y = random.randint(0, max(0, H - crop_size))
        
        return img.crop((x, y, x + crop_size, y + crop_size)), mask.crop((x, y, x + crop_size, y + crop_size))
        '''
        return img, mask
    else:
        # 随机选择一个非背景连通区域（1..num_labels-1）
        # chosen_label = random.randint(1, num_labels - 1)

        # 该区域对应的矩形框信息 stats 中格式为：[left, top, width, height, area]
        # left, top, width, height, area = stats[chosen_label]
        
        areas = stats[1:, 4]
        max_idx = np.argmax(areas) + 1  # +1 因为跳过背景区域0

        left, top, width, height, area = stats[max_idx]

        # 找出最长边
        longest_side = max(width, height)

        # 定义放大倍率：2 ~ 3 的随机倍数
        scale_factor = random.uniform(4.0, 8.0)
        
        # 计算放大后的裁剪尺寸（正方形，边长为 longest_side * scale_factor）
        crop_size = max(int(longest_side * scale_factor), min_crop_size)
        crop_size = min(crop_size, min(W, H))  # 限制不能超出图像短边大小（也可以允许非正方形，这里为简化用正方形）
        #print(scale_factor, crop_size)
        # 计算放大框中心（矩形中心）
        cx = left + width // 2
        cy = top + height // 2

        # 计算裁剪框左上角坐标，使裁剪框以 (cx, cy) 为中心
        x = cx - crop_size // 2
        y = cy - crop_size // 2

        # 保证裁剪框在图像范围内
        x = max(0, min(x, W - crop_size))
        y = max(0, min(y, H - crop_size))

        return img.crop((x, y, x + crop_size, y + crop_size)), mask.crop((x, y, x + crop_size, y + crop_size))


def crop_around_mask_center_with_jitter(
    image_pil: Image.Image,
    mask_pil: Image.Image,
    divisors: Union[List[int], Tuple[int, ...]] = (2, 3),
    jitter_fraction: float = 0.33
) -> Tuple[Image.Image, Image.Image, Tuple[int, int, int, int]]:
    """
    以掩码区域的中心为基准进行裁剪，并加入少量随机扰动。
    裁剪尺寸为原始图像尺寸的 1/n。

    参数:
    - image_pil (PIL.Image.Image): 输入的原始图像。
    - mask_pil (PIL.Image.Image): 对应的掩码。
    - divisors (list or tuple): 从中随机选择一个除数 n，用于确定裁剪尺寸 (1/n)。
    - jitter_fraction (float): 扰动范围，占掩码区域尺寸的比例。
      例如 0.33 表示扰动范围是掩码宽度/高度的 1/3。

    返回:
    - tuple: (cropped_image, cropped_mask, final_bbox)
        - cropped_image (PIL.Image.Image): 裁剪后的图像。
        - cropped_mask (PIL.Image.Image): 裁剪后的掩码。
        - final_bbox (tuple): 最终在原图上使用的裁剪框 (left, top, right, bottom)。
    """
    # 1. 输入验证
    if image_pil.size != mask_pil.size:
        raise ValueError("图像和掩码的尺寸必须相同。")

    original_width, original_height = image_pil.size

    # 2. 找到 Mask 的边界框
    mask_bbox = mask_pil.getbbox()
    
    # 3. 计算 Mask 的中心点和尺寸
    m_left, m_top, m_right, m_bottom = mask_bbox
    mask_width = m_right - m_left
    mask_height = m_bottom - m_top
    mask_center_x = m_left + mask_width / 2
    mask_center_y = m_top + mask_height / 2

    # 4. 计算扰动范围并生成随机扰动
    max_jitter_x = mask_width * jitter_fraction
    max_jitter_y = mask_height * jitter_fraction
    
    jitter_x = random.uniform(-max_jitter_x / 2, max_jitter_x / 2)
    jitter_y = random.uniform(-max_jitter_y / 2, max_jitter_y / 2)

    # 5. 计算扰动后的新中心点
    perturbed_center_x = mask_center_x + jitter_x
    perturbed_center_y = mask_center_y + jitter_y

    # 6. 确定裁剪尺寸
    chosen_divisor = random.choice(divisors)
    crop_width = original_width // chosen_divisor
    crop_height = original_height // chosen_divisor

    # 7. 计算裁剪框的左上角坐标
    left = int(perturbed_center_x - crop_width / 2)
    top = int(perturbed_center_y - crop_height / 2)

    # 8. 边界检查，确保裁剪框在图像范围内
    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if left + crop_width > original_width:
        left = original_width - crop_width
    if top + crop_height > original_height:
        top = original_height - crop_height
        
    # 9. 定义最终裁剪框并执行裁剪
    right = left + crop_width
    bottom = top + crop_height
    final_bbox = (left, top, right, bottom)

    cropped_image = image_pil.crop(final_bbox)
    cropped_mask = mask_pil.crop(final_bbox)

    return cropped_image, cropped_mask, final_bbox

def dilate_and_inpaint_region_pil(
    image_pil: Image.Image, 
    mask_pil: Image.Image, 
    kernel_range: Union[int, Tuple[int, int]] = (10, 100), 
    iterations: int = 1
) -> tuple[Image.Image, Image.Image]:
    """
    对给定图像的指定 mask 区域进行膨胀，并使用图像修复技术填充新区域。
    此版本接受并返回 PIL.Image 对象。

    参数:
    - image_pil (PIL.Image.Image): 输入的原始图像。
    - mask_pil (PIL.Image.Image): 输入的二值掩码，白色区域代表要处理的区域。
    - kernel_size (int): 用于膨胀的内核大小，值越大，膨胀范围越广。
    - iterations (int): 膨胀操作的迭代次数，次数越多，膨胀范围越广。

    返回:
    - tuple: (inpainted_image_pil, dilated_mask_pil)
        - inpainted_image_pil (PIL.Image.Image): 区域膨胀并修复后的新图像。
        - dilated_mask_pil (PIL.Image.Image): 膨胀后的新掩码。
    """
    # 1. [PIL -> OpenCV] 将 PIL 图像转换为 OpenCV 使用的 NumPy 数组格式
    #    - 主图像: PIL (RGB) -> NumPy -> OpenCV (BGR)
    #    - 掩码: PIL (L) -> NumPy
    min_k, max_k = kernel_range
        
    kernel_size = np.random.randint(min_k, max_k)
    image_np = np.array(image_pil.convert("RGB"))
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    mask_np = np.array(mask_pil.convert("L"))

    # --- 以下是您提供的核心 OpenCV 逻辑，几乎不变 ---
    
    # 2. 确保 mask 是二值的 (0 和 255)
    _, mask_binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

    # 3. 创建膨胀操作的内核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # 4. 对 mask 进行膨胀操作
    dilated_mask_np = cv2.dilate(mask_binary, kernel, iterations=iterations)

    # 5. 计算需要修复的区域（膨胀后的 mask 减去原始 mask）
    inpaint_mask_np = cv2.subtract(dilated_mask_np, mask_binary)

    # 6. 使用 cv2.inpaint 进行图像修复
    inpainted_image_cv = cv2.inpaint(image_cv, inpaint_mask_np, inpaintRadius=kernel_size//2, flags=cv2.INPAINT_TELEA)

    # --- 核心逻辑结束 ---

    # 7. [OpenCV -> PIL] 将处理结果转换回 PIL.Image 对象
    #    - 主图像: OpenCV (BGR) -> NumPy (RGB) -> PIL
    #    - 掩码: NumPy -> PIL (L)
    inpainted_image_rgb = cv2.cvtColor(inpainted_image_cv, cv2.COLOR_BGR2RGB)
    inpainted_image_pil = Image.fromarray(inpainted_image_rgb)
    
    dilated_mask_pil = Image.fromarray(dilated_mask_np, mode='L')

    return inpainted_image_pil, dilated_mask_pil


def random_grid_crop(
    image_pil: Image.Image,
    mask_pil: Image.Image,
    divisors: Union[List[int], Tuple[int, ...]] = (2, 3)
) -> Tuple[Image.Image, Image.Image]:
    """
    将图像和掩码切分为 n x n 的网格，并返回包含最大掩码区域的那个网格块。
    此版本使用 NumPy 向量化操作，避免了 for 循环，效率更高。

    参数:
    - image_pil (PIL.Image.Image): 输入的原始图像。
    - mask_pil (PIL.Image.Image): 对应的二值掩码。
    - grid_size (int): 网格的边长 (例如, 2 代表 2x2, 3 代表 3x3)。

    返回:
    - tuple: (cropped_image, cropped_mask)
        - cropped_image (PIL.Image.Image): 包含最大掩码区域的图像块。
        - cropped_mask (PIL.Image.Image): 对应的掩码块。
    """
    # 1. 输入验证

    if image_pil.size != mask_pil.size:
        raise ValueError("图像和掩码的尺寸必须相同。")
    grid_size = random.choice(divisors)
    if grid_size < 1:
        raise ValueError("grid_size 必须是正整数。")

    # 2. 准备工作
    width, height = image_pil.size
    tile_width = width // grid_size
    tile_height = height // grid_size
    
    # 将 PIL Mask 转换为 NumPy 数组 (0-255)
    mask_array = np.array(mask_pil.convert("L"))

    # 3. 核心向量化操作
    # 为了能被 reshape，需要保证尺寸能被整除。我们裁剪一下数组。
    h_new, w_new = grid_size * tile_height, grid_size * tile_width
    mask_cropped = mask_array[:h_new, :w_new]

    # 将 2D 数组 (H, W) 变形为 4D 数组 (grid_h, tile_h, grid_w, tile_w)
    # 然后交换轴，使其变为 (grid_h, grid_w, tile_h, tile_w)
    reshaped_mask = mask_cropped.reshape(grid_size, tile_height, grid_size, tile_width).swapaxes(1, 2)

    # 沿着最后两个维度 (tile_h, tile_w) 求和，得到每个块的掩码面积
    # 结果是一个 (grid_size, grid_size) 的数组
    tile_sums = reshaped_mask.sum(axis=(2, 3))

    # 4. 找到最大面积块的索引
    # np.argmax 返回的是扁平化后数组的索引，需要用 unravel_index 转换回 2D 索引
    max_idx_flat = np.argmax(tile_sums)
    max_row, max_col = np.unravel_index(max_idx_flat, (grid_size, grid_size))

    # 5. 计算最佳块的边界框并裁剪
    left = max_col * tile_width
    top = max_row * tile_height
    right = left + tile_width
    bottom = top + tile_height
    best_tile_bbox = (left, top, right, bottom)

    cropped_image = image_pil.crop(best_tile_bbox)
    cropped_mask = mask_pil.crop(best_tile_bbox)

    return cropped_image, cropped_mask


class Dataset(data.Dataset):
    def __init__(self, root, img_size, train_transform, test_transform, norm_transform, dataset_name, save_dir, mode='train', seed=10, class_name=None):
        self.root = root
        self.img_size = img_size
        self.train_transform = train_transform
        self.test_transform = test_transform      
        self.norm_transform = norm_transform
        self.dataset_name = dataset_name
        self.mode = mode
        self.pseudo_anomaly = False
        self.save_dir = save_dir
        self.mosaic_aug_prob = 0.5
        
        meta_info_json = json.load(open(f'{self.root}/meta.json', 'r'))
        meta_test_info = meta_info_json['test']
        meta_train_info = meta_info_json['train']

        self.prompt_data_all = meta_train_info

        if class_name is not None:
            if isinstance(class_name, list):
                self.cls_names = class_name
            else:
                self.cls_names = [class_name]
        else: 
            self.cls_names = list(meta_test_info.keys())

        self.data_all = []
        for cls_name in self.cls_names:
            '''
            if dataset_name == 'medical-seg':
                self.data_all.extend(meta_test_info[cls_name])
                self.data_all.extend(meta_train_info[cls_name])
            else:
                self.data_all.extend(meta_test_info[cls_name])
            '''
            ################### for tsne visualization ###############
            # # for all anomaly images
            # data_temp = [item for item in meta_test_info[cls_name] if item['anomaly'] == 1]
            # # randomly select up to 50 images
            # sample_size = min(len(data_temp), 50)
            # if sample_size > 0:
            #     data_temp = random.sample(data_temp, sample_size)
            # self.data_all.extend(data_temp)
            ###########################################################
            
            self.data_all.extend(meta_test_info[cls_name])
          
                
        self.data_all_dict = meta_test_info.copy()
        

        self.length = len(self.data_all)
        
        if dataset_name == 'RealIAD':
            self.obj_list = self.cls_names
            self.class_name_map_class_id = {}
            for k, index in zip(self.cls_names, range(len(self.cls_names))):
                self.class_name_map_class_id[k] = index
            self.view_list = ['C1', 'C2', 'C3', 'C4', "C5"]
        elif dataset_name == 'Real-IAD-Variety':
            self.obj_list = self.cls_names
            self.class_name_map_class_id = {}
            for k, index in zip(self.cls_names, range(len(self.cls_names))):
                self.class_name_map_class_id[k] = index
            self.view_list = ['C01', 'C02', 'C03', 'C04', "C05"]
            
        elif dataset_name == 'mvtec':
            self.obj_list = self.cls_names
            self.class_name_map_class_id = {}
            for k, index in zip(self.cls_names, range(len(self.cls_names))):
                self.class_name_map_class_id[k] = index
            self.view_list = ['0']

        else:
            self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
            self.view_list = ['0']


        
        if self.pseudo_anomaly:
            self.cutpaste_transform = CutPaste()
            self.perlinpaste_transform = PerlinPaste(self.root, self.data_all)
        
        self.boxes = {obj: deque(maxlen=10) for obj in self.obj_list}
        self.random_rotate = A.Rotate(crop_border = True, p=0.5)

    def __len__(self):
        return self.length
    
    def read_img_mask(self, img_path, mask_path, anomaly):
        #img = Image.open(os.path.join(self.root, img_path))

        try:
            img = Image.open(os.path.join(self.root, img_path))
        except UnidentifiedImageError:
            img = tiff.imread(os.path.join(self.root, img_path))
            '''
            # If the TIFF is multi‑page, you may get a shape like (pages, H, W, C).
            # For most datasets we only need the first page.
            if img.ndim == 4:          # (pages, H, W, C)
                img = img[0]           # take the first page
            elif img.ndim == 3 and img.shape[0] <= 4 and img.shape[0] in {1, 3, 4}:
                # Some TIFFs store channels as the first dimension (C, H, W)
                img = np.moveaxis(img, 0, -1)   # -> (H, W, C)

            # Ensure the array is in a format Pillow understands:
            #   * uint8  -> normal 8‑bit image
            #   * uint16 -> Pillow can handle it, but you may want to scale to 8‑bit
            #   * float32 -> also OK, Pillow will treat it as 0‑255 after clipping
            if img.dtype == np.uint16:
                # Optional: scale 16‑bit to 8‑bit (lossless if you need the full range)
                img = (img / 256).astype(np.uint8)
            '''
            img = Image.fromarray(img)


        if img.mode == 'L':
            img = Image.merge("RGB", (img.copy(), img.copy(), img.copy()))
        elif img.mode == 'RGBA':
            img = img.convert('RGB')

        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                # just for classification not report error
                img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')

        if img_mask.size != img.size:
            img = img.resize((img_mask.size))
            print(f"Warning: Mask size mismatch for {img_path}. ")
            #import ipdb; ipdb.set_trace()

        
        return img, img_mask


    def __getitem__(self, index):

        # query image
        data = self.data_all[index]

        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
        view_id = data['view_id'] if 'view_id' in data else '0'
        img, img_mask = self.read_img_mask(img_path, mask_path, anomaly)

        # grid crop on visa dataset
        if self.mode == 'train' and self.dataset_name == 'visa' and torch.rand(1) > self.mosaic_aug_prob:
            if anomaly == 1:
                img, img_mask, final_box = crop_around_mask_center_with_jitter(img, img_mask, divisors=(2, 3))
                self.boxes[cls_name].append(final_box)
            elif len(self.boxes[cls_name]) >= 1:
                # final
                final_bbox = random.choice(self.boxes[cls_name])
                img, img_mask = img.crop(final_bbox), img_mask.crop(final_bbox)
                
                
                '''
                # baseline
                #img, img_mask, final_box = crop_around_mask_center_with_jitter(img, img_mask, divisors=(2, 3))
                img, img_mask = random_grid_crop(img, img_mask, divisors=(2, 3))
                '''

        
        # grip mosaic on mvtec dataset
        if self.mode == 'train' and self.dataset_name == 'mvtec' and np.random.uniform() > self.mosaic_aug_prob:
            random.shuffle(self.data_all_dict[cls_name])
            num_bgs = np.random.choice([2**2, 3**2], p=[0.5, 0.5]) #
            if anomaly == 0:
                # final
                normal_data_list = [data_item for data_item in self.data_all_dict[cls_name] if data_item['anomaly']==0]
                select_data = np.random.choice(normal_data_list, size=num_bgs-1, replace=True)

                # baseline
                # normal_data_list = [data_item for data_item in self.data_all if data_item['anomaly']==0]
                # select_data = np.random.choice(normal_data_list, size=num_bgs-1, replace=True)
            else:
                # final
                select_data = np.random.choice(self.data_all_dict[cls_name], size=num_bgs-1, replace=True)

                # baseline
                # select_data = np.random.choice(self.data_all, size=num_bgs-1, replace=True)

            select_imgs, select_masks, select_labels = [img], [img_mask], [anomaly]
            for data_item in select_data:
                select_img_path, select_mask_path, select_cls_name, select_specie_name, select_anomaly = data_item['img_path'], data_item['mask_path'], data_item['cls_name'], \
                                                                data_item['specie_name'], data_item['anomaly']
                
                select_img, select_mask = self.read_img_mask(select_img_path, select_mask_path, select_anomaly)
                select_aug_data = self.random_rotate(image=np.array(select_img), mask=np.array(select_mask))
                select_img, select_mask = Image.fromarray(select_aug_data['image']), Image.fromarray(select_aug_data['mask'])
                
                '''
                if torch.rand(1) > 0.5:
                    #select_img, select_mask = random_crop_with_mask_pil(select_img, select_mask)
                    #select_img, select_mask = dilate_and_inpaint_region_pil(select_img, select_mask)
                    if select_anomaly == 1:
                        select_img, select_mask, final_box = crop_around_mask_center_with_jitter(select_img, select_mask, divisors=(2, 3))
                        self.boxes[cls_name].append(final_box)
                    elif len(self.boxes[cls_name]) >= 1:
                        final_bbox = random.choice(self.boxes[cls_name])
                        select_img, select_mask = select_img.crop(final_bbox), select_mask.crop(final_bbox)
                '''
                select_imgs.append(select_img)
                select_masks.append(select_mask)
                select_labels.append(select_anomaly)

            img, img_mask = mosaic_aug(select_imgs, select_masks, img_size=self.img_size)
            
        if self.mode == 'train':
           aug_data = self.train_transform(image=np.array(img), mask=np.array(img_mask))
        else:
            try:
                aug_data = self.test_transform(image=np.array(img), mask=np.array(img_mask))
            except UnidentifiedImageError:
                print(img_path, img.size, img_mask.size)
        
        aug_data = self.norm_transform(image = aug_data['image'], mask = aug_data['mask'])

        img, img_mask = aug_data['image'], aug_data['mask']
        img_mask = img_mask.float() / 255
    
        return {'query_img': img, 'query_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly, 
                'view_id': view_id,
                'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name]}    

class PromptDataset(data.Dataset):
    def __init__(self, root, img_size, train_transform, test_transform, norm_transform, dataset_name, k_shots, save_dir, mode='test', seed=10, class_name=None):
        self.root = root
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.norm_transform = norm_transform
        self.k_shots = k_shots
        self.mode = mode
        self.save_dir = save_dir
        self.prompt_data_all = []

        meta_info_json = json.load(open(f'{self.root}/meta.json', 'r'))

        if k_shots >= 0:
            if meta_info_json['train']:
                meta_train_info = meta_info_json['train']
            else:
                meta_train_info = meta_info_json['test']
        elif k_shots == -1:
            meta_train_info = meta_info_json['test']
        

        if class_name is not None:
            if isinstance(class_name, list):
                self.cls_names = class_name
            else:
                self.cls_names = [class_name]
        else: 
            self.cls_names = list(meta_train_info.keys())
        
        if dataset_name == 'RealIAD':
            self.obj_list = self.cls_names
            self.class_name_map_class_id = {}
            for k, index in zip(self.cls_names, range(len(self.cls_names))):
                self.class_name_map_class_id[k] = index
            self.view_list = ['C1', 'C2', 'C3', 'C4', "C5"]
        elif dataset_name == 'Real-IAD-Variety':
            self.obj_list = self.cls_names
            self.class_name_map_class_id = {}
            for k, index in zip(self.cls_names, range(len(self.cls_names))):
                self.class_name_map_class_id[k] = index
            self.view_list = ['C01', 'C02', 'C03', 'C04', "C05"]
        else:
            self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
            self.view_list = ['0']

        
        if self.k_shots > 0:
            #fs_txt = f'{dataset_name}_{seed}seed_{str(self.k_shots)}shot_{mode}_prompts.txt'
            #prompt_save_dir = os.path.join(save_dir, fs_txt)
            fs_json = f'{dataset_name}_{seed}seed_{str(self.k_shots)}shot_{mode}_prompts.json'
            prompt_save_dir = os.path.join(save_dir, fs_json)

            prompt_json = {}

            if len(self.view_list) > 1:
                for cls_name in self.cls_names:
                    data_tmp = meta_train_info[cls_name]
                    prompt_json[cls_name] = {}
                    for view_id in self.view_list:
                        torch.manual_seed(seed)
                        prompt_json[cls_name][view_id] = []
                        data_view_tmp = [item for item in data_tmp if item['view_id'] == view_id]
                        indices = torch.randint(0, len(data_view_tmp), (self.k_shots,))

                        prompt_json[cls_name][view_id] = [data_view_tmp[i] for i in indices]

                        self.prompt_data_all.extend([data_view_tmp[i] for i in indices])
                        #for i in range(len(indices)):
                        #    with open(prompt_save_dir, "a") as f:
                        #        f.write(data_view_tmp[indices[i]]['img_path'] + '\n')
            

            else:
                for cls_name in self.cls_names:
                    data_tmp = meta_train_info[cls_name]
                    prompt_json[cls_name] = {}
                    #data_tmp = [item for item in data_tmp if item['anomaly'] == 1] # 排除OK
                    torch.manual_seed(seed)
                    indices = torch.randint(0, len(data_tmp), (self.k_shots,))
                    prompt_json[cls_name] = [data_tmp[i] for i in indices]
                    self.prompt_data_all.extend([data_tmp[i] for i in indices])
                    
                    '''
                    for i in range(len(indices)):
                        with open(prompt_save_dir, "a") as f:
                            f.write(data_tmp[indices[i]]['img_path'] + '\n')
                    '''
            if os.path.exists(prompt_save_dir):
                with open(prompt_save_dir, 'r', encoding='utf-8') as f:
                    prompts = json.load(f)
            else:
                prompts = {}
            
            # 追加新数据
            prompts.update(prompt_json)
            
            with open(prompt_save_dir, 'w') as f:
                f.write(json.dumps(prompts, indent=4) + "\n")
        
        elif k_shots == -1:
            print("全部测试数据无论是OK还是NG,都来用作构建MemoryBank")
            for cls_name in self.cls_names:
                data_tmp = meta_train_info[cls_name]
                self.prompt_data_all.extend(data_tmp)


        self.length = len(self.prompt_data_all)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # query image
        data = self.prompt_data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
        view_id = data['view_id'] if 'view_id' in data else '0'
        #img = Image.open(os.path.join(self.root, img_path))
        try:
            img = Image.open(os.path.join(self.root, img_path))
        except UnidentifiedImageError:
            img = tiff.imread(os.path.join(self.root, img_path))
            img = Image.fromarray(img)

        if img.mode == 'L':
            # 如果是灰度图像，直接复制三份
            img = Image.merge("RGB", (img.copy(), img.copy(), img.copy()))
        elif img.mode == 'RGBA':
            img = img.convert('RGB')

        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                # just for classification not report error
                img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        
        # transforms
        if self.mode == 'train':
           trans_data = self.train_transform(image=np.array(img), mask=np.array(img_mask))
        else:
           trans_data = self.test_transform(image=np.array(img), mask=np.array(img_mask))
        trans_data = self.norm_transform(image = trans_data['image'], mask = trans_data['mask'])
        img, img_mask = trans_data['image'], trans_data['mask']
        img_mask = img_mask.float() / 255

        return {'prompt_img': img, 'prompt_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
                'view_id': view_id,
                'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name]}  