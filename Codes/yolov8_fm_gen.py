from ultralytics import YOLO
import torch

# Load a model
yolo_model = YOLO('yolov8n-seg.pt')  # load an official model

global_index = 0
global_feature_maps_l0 = []
global_feature_maps_l1 = []
global_feature_maps_l2 = []

def hook_fn_before_seg_head_l0(m, x, y):
    # In the case of multiple images it should return all
    for it in y:
        global_feature_maps_l0.append(it.detach())

def hook_fn_before_seg_head_l1(m, x, y):
    # In the case of multiple images it should return all
    for it in y:
        global_feature_maps_l1.append(it.detach())

def hook_fn_before_seg_head_l2(m, x, y):
    # In the case of multiple images it should return all
    for it in y:
        global_feature_maps_l2.append(it.detach())


def hook_install(layer, hook_fn):
    # Register the hook on the 5th layer of the model's 'layer1'
    layer.register_forward_hook(hook_fn)

    # Now, whenever you make a forward pass through the model, the hook will be triggered when the input passes through the 5th layer of 'layer1'.

hook_install(yolo_model.model.model[21], hook_fn_before_seg_head_l2)
hook_install(yolo_model.model.model[18], hook_fn_before_seg_head_l1)
hook_install(yolo_model.model.model[15], hook_fn_before_seg_head_l0)

filen_name = 'test'
# Predict with the model
results = yolo_model(f'{filen_name}.png')

global_feature_maps_l0 = torch.stack(global_feature_maps_l0)
global_feature_maps_l1 = torch.stack(global_feature_maps_l1)
global_feature_maps_l2 = torch.stack(global_feature_maps_l2)

torch.save(global_feature_maps_l0, f'{filen_name}_fm_l0.pt')
torch.save(global_feature_maps_l1, f'{filen_name}_fm_l1.pt')
torch.save(global_feature_maps_l2, f'{filen_name}_fm_l2.pt')
