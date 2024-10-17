import sys
import json
from typing import Optional
import cv2
import torch
import argparse
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from transformers import BlipProcessor, BlipForConditionalGeneration
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from cog import BasePredictor, Input, Path, BaseModel
sys.path.insert(0, "scripts")
import torch.distributed as dist
import os
import pycocotools.mask as maskUtils
import torch.multiprocessing as mp

def oneformer_coco_segmentation(image, oneformer_coco_processor, oneformer_coco_model, rank):
    inputs = oneformer_coco_processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(rank)
    outputs = oneformer_coco_model(**inputs)
    predicted_semantic_map = oneformer_coco_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]])[0]
    return predicted_semantic_map

def oneformer_ade20k_segmentation(image, oneformer_ade20k_processor, oneformer_ade20k_model, rank):
    inputs = oneformer_ade20k_processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(rank)
    outputs = oneformer_ade20k_model(**inputs)
    predicted_semantic_map = oneformer_ade20k_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]])[0]
    return predicted_semantic_map

def oneformer_cityscapes_segmentation(image, oneformer_cityscapes_processor, oneformer_cityscapes_model, rank):
    inputs = oneformer_cityscapes_processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(rank)
    outputs = oneformer_cityscapes_model(**inputs)
    predicted_semantic_map = oneformer_cityscapes_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]])[0]
    return predicted_semantic_map

oneformer_func = {
    'ade20k': oneformer_ade20k_segmentation,
    'coco': oneformer_coco_segmentation,
    'cityscapes': oneformer_cityscapes_segmentation,
    'foggy_driving': oneformer_cityscapes_segmentation
}


def main(rank, args):
    # dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    
    sam = sam_model_registry["vit_h"](checkpoint=args.ckpt_path).to(rank)

    mask_branch_model = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=128 if args.dataset == 'foggy_driving' else 64,
        # Foggy driving (zero-shot evaluate) is more challenging than other dataset, so we use a larger points_per_side
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
        output_mode='coco_rle',
    )
    print('[Model loaded] Mask branch (SAM) is loaded.')
    # yoo can add your own semantic branch here, and modify the following code
    if args.model == 'oneformer':
        from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        if args.dataset == 'ade20k':
            semantic_branch_processor = OneFormerProcessor.from_pretrained(
                "shi-labs/oneformer_ade20k_swin_large")
            semantic_branch_model = OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_ade20k_swin_large").to(rank)
        elif args.dataset == 'cityscapes':
            semantic_branch_processor = OneFormerProcessor.from_pretrained(
                "shi-labs/oneformer_cityscapes_swin_large")
            semantic_branch_model = OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_cityscapes_swin_large").to(rank)
        elif args.dataset == 'foggy_driving':
            semantic_branch_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_dinat_large")
            semantic_branch_model = OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_cityscapes_dinat_large").to(rank)
        else:
            raise NotImplementedError()
    elif args.model == 'segformer':
        from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
        if args.dataset == 'ade20k':
            semantic_branch_processor = SegformerFeatureExtractor.from_pretrained(
                "nvidia/segformer-b5-finetuned-ade-640-640")
            semantic_branch_model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b5-finetuned-ade-640-640").to(rank)
        elif args.dataset == 'cityscapes' or args.dataset == 'foggy_driving':
            semantic_branch_processor = SegformerFeatureExtractor.from_pretrained(
                "nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
            semantic_branch_model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b5-finetuned-cityscapes-1024-1024").to(rank)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    print('[Model loaded] Semantic branch (your own segmentor) is loaded.')
    if args.dataset == 'ade20k':
        filenames = [fn_.replace('.jpg', '') for fn_ in os.listdir(args.data_dir) if '.jpg' in fn_]
    elif args.dataset == 'cityscapes' or args.dataset == 'foggy_driving':
        sub_folders = [fn_ for fn_ in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, fn_))]
        filenames = []
        for sub_folder in sub_folders:
            filenames += [os.path.join(sub_folder, fn_.replace('.png', '')) for fn_ in os.listdir(args.data_dir + sub_folder) if '.png' in fn_]
    #local_filenames = filenames[(len(filenames) // args.world_size + 1) * rank : (len(filenames) // args.world_size + 1) * (rank + 1)]
    local_filenames=filenames
    print('[Image name loaded] get image filename list.')
    print('[SSA start] model inference starts.')

    for i, file_name in enumerate(local_filenames):
        print('[Runing] ', i, '/', len(local_filenames), ' ', file_name, ' on rank ', rank, '/', args.world_size)
        img = img_load(args.data_dir, file_name, args.dataset)
        if args.dataset == 'ade20k':
            id2label = ADE_20K_CONFIG
        # elif args.dataset == 'cityscapes' or args.dataset == 'foggy_driving':
        #     id2label = CONFIG_CITYSCAPES_ID2LABEL
        else:
            raise NotImplementedError()
        with torch.no_grad():
            semantic_segment_anything_inference(file_name, args.out_dir, rank, img=img, save_img=args.save_img,
                                   semantic_branch_processor=semantic_branch_processor,
                                   semantic_branch_model=semantic_branch_model,
                                   mask_branch_model=mask_branch_model,
                                   dataset=args.dataset,
                                   id2label=id2label,
                                   model=args.model)
        # torch.cuda.empty_cache()
    if args.eval and rank==0:
        assert args.gt_path is not None
        eval_pipeline(args.gt_path, args.out_dir, args.dataset)

def parse_args():
    parser = argparse.ArgumentParser(description='Semantically segment anything.')
    parser.add_argument('--data_dir', default='/weka/datasets/XC_Data/openimages_sample',help='specify the root path of images and masks')
    parser.add_argument('--ckpt_path', default='ckp/sam_vit_h_4b8939.pth', help='specify the root path of SAM checkpoint')
    parser.add_argument('--out_dir', default='/weka/datasets/XC_Data/ssa',help='the dir to save semantic annotations')
    parser.add_argument('--save_img', default=False, action='store_true', help='whether to save annotated images')
    parser.add_argument('--world_size', type=int, default=0, help='number of nodes')
    parser.add_argument('--dataset', type=str, default='ade20k', choices=['ade20k', 'cityscapes', 'foggy_driving'], help='specify the set of class names')
    parser.add_argument('--eval', default=False, action='store_true', help='whether to execute evalution')
    parser.add_argument('--gt_path', default=None, help='specify the path to gt annotations')
    parser.add_argument('--model', type=str, default='oneformer', choices=['oneformer', 'segformer'], help='specify the semantic branch model')
    args = parser.parse_args()
    return args

def img_load(data_path, filename, dataset):
    # load image
    if dataset == 'ade20k':
        img = cv2.imread(os.path.join(data_path, filename+'.jpg'))
    elif dataset == 'cityscapes' or dataset == 'foggy_driving':
        img = cv2.imread(os.path.join(data_path, filename+'.png'))
    else:
        raise NotImplementedError()
    return img

def semantic_segment_anything_inference(filename, output_path, rank, img=None, save_img=False,
                                 semantic_branch_processor=None,
                                 semantic_branch_model=None,
                                 mask_branch_model=None,
                                 dataset=None,
                                 id2label=None,
                                 model='segformer'):

    anns = {'annotations': mask_branch_model.generate(img)}
    h, w, _ = img.shape
    class_names = []
    if model == 'oneformer':
        class_ids = oneformer_func[dataset](Image.fromarray(img), semantic_branch_processor,
                                                                        semantic_branch_model, rank)
    # elif model == 'segformer':
    #     class_ids = segformer_func(img, semantic_branch_processor, semantic_branch_model, rank)
    else:
        raise NotImplementedError()
    semantc_mask = class_ids.clone()
    anns['annotations'] = sorted(anns['annotations'], key=lambda x: x['area'], reverse=True)
    for ann in anns['annotations']:
        valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
        # get the class ids of the valid pixels
        propose_classes_ids = class_ids[valid_mask]
        num_class_proposals = len(torch.unique(propose_classes_ids))
        if num_class_proposals == 1:
            semantc_mask[valid_mask] = propose_classes_ids[0]
            ann['class_name'] = id2label['id2label'][str(propose_classes_ids[0].item())]
            ann['class_proposals'] = id2label['id2label'][str(propose_classes_ids[0].item())]
            class_names.append(ann['class_name'])
            # bitmasks.append(maskUtils.decode(ann['segmentation']))
            continue
        top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
        top_1_propose_class_names = [id2label['id2label'][str(class_id.item())] for class_id in top_1_propose_class_ids]

        semantc_mask[valid_mask] = top_1_propose_class_ids
        ann['class_name'] = top_1_propose_class_names[0]
        ann['class_proposals'] = top_1_propose_class_names[0]
        class_names.append(ann['class_name'])
        # bitmasks.append(maskUtils.decode(ann['segmentation']))

        del valid_mask
        del propose_classes_ids
        del num_class_proposals
        del top_1_propose_class_ids
        del top_1_propose_class_names
    
    sematic_class_in_img = torch.unique(semantc_mask)
    semantic_bitmasks, semantic_class_names = [], []

    # semantic prediction
    anns['semantic_mask'] = {}
    for i in range(len(sematic_class_in_img)):
        class_name = id2label['id2label'][str(sematic_class_in_img[i].item())]
        class_mask = semantc_mask == sematic_class_in_img[i]
        class_mask = class_mask.cpu().numpy().astype(np.uint8)
        semantic_class_names.append(class_name)
        semantic_bitmasks.append(class_mask)
        anns['semantic_mask'][str(sematic_class_in_img[i].item())] = maskUtils.encode(np.array((semantc_mask == sematic_class_in_img[i]).cpu().numpy(), order='F', dtype=np.uint8))
        anns['semantic_mask'][str(sematic_class_in_img[i].item())]['counts'] = anns['semantic_mask'][str(sematic_class_in_img[i].item())]['counts'].decode('utf-8')
    
    # if save_img:
    #     imshow_det_bboxes(img,
    #                         bboxes=None,
    #                         labels=np.arange(len(sematic_class_in_img)),
    #                         segms=np.stack(semantic_bitmasks),
    #                         class_names=semantic_class_names,
    #                         font_size=25,
    #                         show=False,
    #                         out_file=os.path.join(output_path, filename + '_semantic.png'))
    #     print('[Save] save SSA prediction: ', os.path.join(output_path, filename + '_semantic.png'))
    draw_bboxes_and_masks(img, bboxes=None, labels=np.arange(len(sematic_class_in_img)), segms=semantic_bitmasks, class_names=semantic_class_names, font_size=25, out_file=os.path.join(output_path, filename + '_semantic.png'))
    # 手动清理不再需要的变量
    del img
    del anns
    del class_ids
    del semantc_mask
    # del bitmasks
    del class_names
    del semantic_bitmasks
    del semantic_class_names

    # gc.collect()
def draw_bboxes_and_masks(img, bboxes, labels, segms, class_names, font_size, out_file):
    # Draw bounding boxes
    if bboxes is not None:
        for i, bbox in enumerate(bboxes):
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(img, class_names[labels[i]], (int(x_min), int(y_min) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_size/10, (255, 0, 0), 2)
    
    # Apply transparent mask overlays (light green)
    if segms is not None:
        mask_color = np.array([0, 255, 0], dtype=np.uint8)  # Light green
        alpha = 0.4  # Transparency factor (0 = fully transparent, 1 = opaque)
        
        for mask in segms:
            # Create a mask image with the color and blend it with the original image
            colored_mask = np.zeros_like(img, dtype=np.uint8)
            colored_mask[mask > 0] = mask_color
            
            # Blend the colored mask with the original image
            img = cv2.addWeighted(colored_mask, alpha, img, 1 - alpha, 0)

    # Save the output
    cv2.imwrite(out_file, img)
    print(f"Saved to {out_file}")

COCO_CONFIG = {"id2label": {
    "0": "person",
    "1": "bicycle",
    "2": "car",
    "3": "motorcycle",
    "4": "airplane",
    "5": "bus",
    "6": "train",
    "7": "truck",
    "8": "boat",
    "9": "traffic light",
    "10": "fire hydrant",
    "11": "stop sign",
    "12": "parking meter",
    "13": "bench",
    "14": "bird",
    "15": "cat",
    "16": "dog",
    "17": "horse",
    "18": "sheep",
    "19": "cow",
    "20": "elephant",
    "21": "bear",
    "22": "zebra",
    "23": "giraffe",
    "24": "backpack",
    "25": "umbrella",
    "26": "handbag",
    "27": "tie",
    "28": "suitcase",
    "29": "frisbee",
    "30": "skis",
    "31": "snowboard",
    "32": "sports ball",
    "33": "kite",
    "34": "baseball bat",
    "35": "baseball glove",
    "36": "skateboard",
    "37": "surfboard",
    "38": "tennis racket",
    "39": "bottle",
    "40": "wine glass",
    "41": "cup",
    "42": "fork",
    "43": "knife",
    "44": "spoon",
    "45": "bowl",
    "46": "banana",
    "47": "apple",
    "48": "sandwich",
    "49": "orange",
    "50": "broccoli",
    "51": "carrot",
    "52": "hot dog",
    "53": "pizza",
    "54": "donut",
    "55": "cake",
    "56": "chair",
    "57": "couch",
    "58": "potted plant",
    "59": "bed",
    "60": "dining table",
    "61": "toilet",
    "62": "tv",
    "63": "laptop",
    "64": "mouse",
    "65": "remote",
    "66": "keyboard",
    "67": "cell phone",
    "68": "microwave",
    "69": "oven",
    "70": "toaster",
    "71": "sink",
    "72": "refrigerator",
    "73": "book",
    "74": "clock",
    "75": "vase",
    "76": "scissors",
    "77": "teddy bear",
    "78": "hair drier",
    "79": "toothbrush",
    "80": "banner",
    "81": "blanket",
    "82": "bridge",
    "83": "cardboard",
    "84": "counter",
    "85": "curtain",
    "86": "door-stuff",
    "87": "floor-wood",
    "88": "flower",
    "89": "fruit",
    "90": "gravel",
    "91": "house",
    "92": "light",
    "93": "mirror-stuff",
    "94": "net",
    "95": "pillow",
    "96": "platform",
    "97": "playingfield",
    "98": "railroad",
    "99": "river",
    "100": "road",
    "101": "roof",
    "102": "sand",
    "103": "sea",
    "104": "shelf",
    "105": "snow",
    "106": "stairs",
    "107": "tent",
    "108": "towel",
    "109": "wall-brick",
    "110": "wall-stone",
    "111": "wall-tile",
    "112": "wall-wood",
    "113": "water-other",
    "114": "window-blind",
    "115": "window-other",
    "116": "tree-merged",
    "117": "fence-merged",
    "118": "ceiling-merged",
    "119": "sky-other-merged",
    "120": "cabinet-merged",
    "121": "table-merged",
    "122": "floor-other-merged",
    "123": "pavement-merged",
    "124": "mountain-merged",
    "125": "grass-merged",
    "126": "dirt-merged",
    "127": "paper-merged",
    "128": "food-other-merged",
    "129": "building-other-merged",
    "130": "rock-merged",
    "131": "wall-other-merged",
    "132": "rug-merged"
},
"refined_id2label": {
    "0": "person",
    "1": "bicycle",
    "2": "car",
    "3": "motorcycle",
    "4": "airplane",
    "5": "bus",
    "6": "train",
    "7": "truck",
    "8": "boat",
    "9": "traffic light",
    "10": "fire hydrant",
    "11": "stop sign",
    "12": "parking meter",
    "13": "bench",
    "14": "bird",
    "15": "cat",
    "16": "dog",
    "17": "horse",
    "18": "sheep",
    "19": "cow",
    "20": "elephant",
    "21": "bear",
    "22": "zebra",
    "23": "giraffe",
    "24": "backpack",
    "25": "umbrella",
    "26": "handbag",
    "27": "tie",
    "28": "suitcase",
    "29": "frisbee",
    "30": "skis",
    "31": "snowboard",
    "32": "sports ball",
    "33": "kite",
    "34": "baseball bat",
    "35": "baseball glove",
    "36": "skateboard",
    "37": "surfboard",
    "38": "tennis racket",
    "39": "bottle",
    "40": "wine glass",
    "41": "cup",
    "42": "fork",
    "43": "knife",
    "44": "spoon",
    "45": "bowl",
    "46": "banana",
    "47": "apple",
    "48": "sandwich",
    "49": "orange",
    "50": "broccoli",
    "51": "carrot",
    "52": "hot dog",
    "53": "pizza",
    "54": "donut",
    "55": "cake",
    "56": "chair",
    "57": "couch",
    "58": "potted plant",
    "59": "bed",
    "60": "dining table",
    "61": "toilet",
    "62": "tv",
    "63": "laptop",
    "64": "mouse",
    "65": "remote",
    "66": "keyboard",
    "67": "cell phone",
    "68": "microwave",
    "69": "oven",
    "70": "toaster",
    "71": "sink",
    "72": "refrigerator",
    "73": "book",
    "74": "clock",
    "75": "vase",
    "76": "scissors",
    "77": "teddy bear",
    "78": "hair drier",
    "79": "toothbrush",
    "80": "banner",
    "81": "blanket",
    "82": "bridge",
    "83": "cardboard",
    "84": "counter",
    "85": "curtain",
    "86": "door",
    "87": "floor-wood",
    "88": "flower",
    "89": "fruit",
    "90": "gravel",
    "91": "house",
    "92": "light",
    "93": "mirror",
    "94": "net",
    "95": "pillow",
    "96": "platform",
    "97": "playingfield",
    "98": "railroad",
    "99": "river",
    "100": "road",
    "101": "roof",
    "102": "sand",
    "103": "sea",
    "104": "shelf",
    "105": "snow",
    "106": "stairs",
    "107": "tent",
    "108": "towel",
    "109": "wall-brick",
    "110": "wall-stone",
    "111": "wall-tile",
    "112": "wall",
    "113": "water",
    "114": "window-blind",
    "115": "window",
    "116": "tree",
    "117": "fence",
    "118": "ceiling",
    "119": "sky",
    "120": "cabinet",
    "121": "table",
    "122": "floor",
    "123": "pavement",
    "124": "mountain",
    "125": "grass",
    "126": "dirt",
    "127": "paper",
    "128": "food",
    "129": "building",
    "130": "rock",
    "131": "wall",
    "132": "rug"
}
}
ADE_20K_CONFIG = {
  "id2label": {
    "0": "wall",
    "1": "building",
    "2": "sky",
    "3": "floor",
    "4": "tree",
    "5": "ceiling",
    "6": "road",
    "7": "bed ",
    "8": "windowpane",
    "9": "grass",
    "10": "cabinet",
    "11": "sidewalk",
    "12": "person",
    "13": "earth",
    "14": "door",
    "15": "table",
    "16": "mountain",
    "17": "plant",
    "18": "curtain",
    "19": "chair",
    "20": "car",
    "21": "water",
    "22": "painting",
    "23": "sofa",
    "24": "shelf",
    "25": "house",
    "26": "sea",
    "27": "mirror",
    "28": "rug",
    "29": "field",
    "30": "armchair",
    "31": "seat",
    "32": "fence",
    "33": "desk",
    "34": "rock",
    "35": "wardrobe",
    "36": "lamp",
    "37": "bathtub",
    "38": "railing",
    "39": "cushion",
    "40": "base",
    "41": "box",
    "42": "column",
    "43": "signboard",
    "44": "chest of drawers",
    "45": "counter",
    "46": "sand",
    "47": "sink",
    "48": "skyscraper",
    "49": "fireplace",
    "50": "refrigerator",
    "51": "grandstand",
    "52": "path",
    "53": "stairs",
    "54": "runway",
    "55": "case",
    "56": "pool table",
    "57": "pillow",
    "58": "screen door",
    "59": "stairway",
    "60": "river",
    "61": "bridge",
    "62": "bookcase",
    "63": "blind",
    "64": "coffee table",
    "65": "toilet",
    "66": "flower",
    "67": "book",
    "68": "hill",
    "69": "bench",
    "70": "countertop",
    "71": "stove",
    "72": "palm",
    "73": "kitchen island",
    "74": "computer",
    "75": "swivel chair",
    "76": "boat",
    "77": "bar",
    "78": "arcade machine",
    "79": "hovel",
    "80": "bus",
    "81": "towel",
    "82": "light",
    "83": "truck",
    "84": "tower",
    "85": "chandelier",
    "86": "awning",
    "87": "streetlight",
    "88": "booth",
    "89": "television receiver",
    "90": "airplane",
    "91": "dirt track",
    "92": "apparel",
    "93": "pole",
    "94": "land",
    "95": "bannister",
    "96": "escalator",
    "97": "ottoman",
    "98": "bottle",
    "99": "buffet",
    "100": "poster",
    "101": "stage",
    "102": "van",
    "103": "ship",
    "104": "fountain",
    "105": "conveyer belt",
    "106": "canopy",
    "107": "washer",
    "108": "plaything",
    "109": "swimming pool",
    "110": "stool",
    "111": "barrel",
    "112": "basket",
    "113": "waterfall",
    "114": "tent",
    "115": "bag",
    "116": "minibike",
    "117": "cradle",
    "118": "oven",
    "119": "ball",
    "120": "food",
    "121": "step",
    "122": "tank",
    "123": "trade name",
    "124": "microwave",
    "125": "pot",
    "126": "animal",
    "127": "bicycle",
    "128": "lake",
    "129": "dishwasher",
    "130": "screen",
    "131": "blanket",
    "132": "sculpture",
    "133": "hood",
    "134": "sconce",
    "135": "vase",
    "136": "traffic light",
    "137": "tray",
    "138": "ashcan",
    "139": "fan",
    "140": "pier",
    "141": "crt screen",
    "142": "plate",
    "143": "monitor",
    "144": "bulletin board",
    "145": "shower",
    "146": "radiator",
    "147": "glass",
    "148": "clock",
    "149": "flag"}
}

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if args.world_size > 1:
        mp.spawn(main,args=(args,),nprocs=args.world_size,join=True)
    else:
        main(0, args)