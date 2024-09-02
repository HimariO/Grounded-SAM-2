import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_camera_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import BoxDictionaryModel, ObjectInfo
import json
import copy

"""
Step 1: Environment settings and model initialization
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
torch.inference_mode().__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t_480.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("config", model_cfg)
print("device", device)

video_predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
# sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
# image_predictor = SAM2ImagePredictor(sam2_image_model)


# init grounding dino model from huggingface
model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = "car."

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`  
video_dir = "notebooks/videos/car"
# 'output_dir' is the directory to save the annotated frames
output_dir = "./outputs"
# 'output_video_path' is the path to save the final video
output_video_path = "./outputs/output.mp4"
# create the output directory
CommonUtils.creat_dirs(output_dir)
mask_data_dir = os.path.join(output_dir, "mask_data")
json_data_dir = os.path.join(output_dir, "json_data")
result_dir = os.path.join(output_dir, "result")
CommonUtils.creat_dirs(mask_data_dir)
CommonUtils.creat_dirs(json_data_dir)
# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# init video predictor state
# inference_state = video_predictor.init_state(video_path=video_dir, offload_video_to_cpu=True, async_loading_frames=True)
step = 200 # the step to sample frames for Grounding DINO predictor

sam2_masks = BoxDictionaryModel()
PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
objects_count = 0


def object_detect(image):
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.25,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]
    )

    # process the detection results
    input_boxes = results[0]["boxes"] # .cpu().numpy()
    # print("results[0]",results[0])
    OBJECTS = results[0]["labels"]
    return input_boxes, OBJECTS


"""
Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for all frames
"""
print("Total frames:", len(frame_names))
for start_frame_idx in range(0, len(frame_names), step):
# prompt grounding dino to get the box coordinates on specific frame
    print("start_frame_idx", start_frame_idx)
    # continue
    img_path = os.path.join(video_dir, frame_names[start_frame_idx])
    image = Image.open(img_path)
    image_base_name = frame_names[start_frame_idx].split(".")[0]
    mask_dict = BoxDictionaryModel(promote_type=PROMPT_TYPE_FOR_VIDEO, box_name=f"mask_{image_base_name}.npy")

    # run Grounding DINO on the image
    input_boxes, OBJECTS = object_detect(image)

    """
    Step 3: Register each object's positive points to video predictor
    """

    # If you are using point prompts, we uniformly sample positive points based on the mask
    if mask_dict.promote_type == "mask":
        mask_dict.add_new_frame_annotation(
            box_list=torch.tensor(input_boxes), 
            label_list=OBJECTS
        )
    else:
        raise NotImplementedError("SAM 2 video predictor only support mask prompts")


    """
    Step 4: Propagate the video predictor to get the segmentation results for each frame
    """
    objects_count = mask_dict.update_boxes(tracking_annotation_dict=sam2_masks, iou_threshold=0.4, objects_count=objects_count)
    print("objects_count", objects_count)
    # video_predictor.reset_state(inference_state)
    if len(mask_dict.labels) == 0:
        print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
        continue

    video_segments = {}  # output the following {step} frames tracking masks

    frame_path = os.path.join(video_dir, frame_names[start_frame_idx])
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_predictor.load_first_frame(frame)
    
    first_frame_masks = BoxDictionaryModel()
    for object_id, object_info in mask_dict.labels.items():
        frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_prompt(
            frame_idx=0,
            obj_id=object_id,
            bbox=[[object_info.x1, object_info.y1], [object_info.x2, object_info.y2]],
        )
        print('add_new_prompt', out_obj_ids)

    # for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
    out_obj_ids = []
    for fid in range(start_frame_idx + 1, min(start_frame_idx + step, len(frame_names))):
        frame_path = os.path.join(video_dir, frame_names[fid])
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if fid % 10 == 0:
            non_cond_frame_outputs = video_predictor.condition_state["output_dict"]['non_cond_frame_outputs']
            non_cond_frame_outputs = {k: v for k, v in non_cond_frame_outputs.items()} # shallow copy
            video_predictor.load_first_frame(frame, frame_idx=fid)
            # video_predictor.condition_state['images'][fid] = frame
            video_predictor.condition_state["output_dict"]['non_cond_frame_outputs'] = non_cond_frame_outputs
            video_predictor.frame_idx = max(non_cond_frame_outputs.keys())

            input_boxes, OBJECTS = object_detect(Image.fromarray(frame))
            
            tmp_dict = BoxDictionaryModel()
            tmp_dict.add_new_frame_annotation(
                box_list=torch.tensor(input_boxes), 
                label_list=OBJECTS
            )
            objects_count = tmp_dict.update_boxes(tracking_annotation_dict=mask_dict, iou_threshold=0.4, objects_count=objects_count)
            mask_dict = tmp_dict
            
            for object_id, object_info in mask_dict.labels.items():
                if object_id in out_obj_ids:
                    continue
                
                frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_prompt(
                    frame_idx=fid,
                    obj_id=object_id,
                    bbox=[[object_info.x1, object_info.y1], [object_info.x2, object_info.y2]],
                )
            print("objects_count", objects_count)
        else:
            out_obj_ids, out_mask_logits = video_predictor.track(frame)

        frame_masks = BoxDictionaryModel()
        for i, out_obj_id in enumerate(out_obj_ids):
            out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
            object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = mask_dict.get_target_class_name(out_obj_id))
            object_info.update_box()
            frame_masks.labels[out_obj_id] = object_info
            image_base_name = frame_names[fid].split(".")[0]
            frame_masks.box_name = f"mask_{image_base_name}.npy"
            frame_masks.image_height = out_mask.shape[-2]
            frame_masks.image_width = out_mask.shape[-1]

        video_segments[fid] = frame_masks
        sam2_masks = copy.deepcopy(frame_masks)

    print("video_segments:", len(video_segments))
    """
    Step 5: save the tracking masks and json files
    """
    for frame_idx, frame_masks_info in video_segments.items():
        mask = frame_masks_info.labels
        mask_img = torch.zeros(frame_masks_info.image_height, frame_masks_info.image_width)
        for obj_id, obj_info in mask.items():
            mask_img[obj_info.y1: obj_info.y2, obj_info.x1: obj_info.x2] = obj_id

        mask_img = mask_img.numpy().astype(np.uint16)
        np.save(os.path.join(mask_data_dir, frame_masks_info.box_name), mask_img)

        json_data = frame_masks_info.to_dict()
        json_data_path = os.path.join(json_data_dir, frame_masks_info.box_name.replace(".npy", ".json"))
        with open(json_data_path, "w") as f:
            json.dump(json_data, f)


"""
Step 6: Draw the results and save the video
"""
CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, result_dir)

create_video_from_images(result_dir, output_video_path, frame_rate=15)