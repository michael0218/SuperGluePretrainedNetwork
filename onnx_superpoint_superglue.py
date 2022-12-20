import torch
from models.superpoint_onnx import SuperPoint
from models.superglue_onnx import SuperGlue

# ==================================================================
#
#        super point
#
# ==================================================================
config = {'superpoint': {'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': 1024}, 'superglue': {'weights': 'indoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}}
superpoint = SuperPoint(config.get('superpoint', {}))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dummy_input = torch.randn(1, 1, 480, 640)#.to(device)
# data = {'image':dummy_input}
pred0 = superpoint(dummy_input)
print(pred0)
print('Converting to ONNX....')
torch.onnx.export(superpoint,         # model being run 
    dummy_input,       # model input (or a tuple for multiple inputs) 
    "./onnx/superpoint_opset16.onnx",       # where to save the model  
    export_params=False,  # store the trained parameter weights inside the model file 
    opset_version=16,    # the ONNX version to export the model to 
    do_constant_folding=True,  # whether to execute constant folding for optimization 
    input_names = ['modelInput'],   # the model's input names 
    output_names = ['modelOutput']) # the model's output names 

print(" ") 
print('Model has been converted to ONNX')



print('Converting to ONNX....')
torch.onnx.export(superpoint,         # model being run 
    dummy_input,       # model input (or a tuple for multiple inputs) 
    "./onnx/superpoint_opset16_dynamic.onnx",       # where to save the model  
    export_params=False,  # store the trained parameter weights inside the model file 
    opset_version=16,    # the ONNX version to export the model to 
    do_constant_folding=True,  # whether to execute constant folding for optimization 
    input_names = ['modelInput'],   # the model's input names 
    output_names = ['modelOutput'], # the model's output names 
    dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                            'modelOutput' : {0 : 'batch_size'}}) 




# ==================================================================
#
#        super glue
#
# ==================================================================
data = {'image0':torch.randn(1, 1, 480, 640), 'image1':torch.randn(1, 1, 480, 640), 'keypoints0':torch.randn(1, 382, 2), 'scores0':torch.randn(1, 382), 'descriptors0':torch.randn(1, 256, 382), 'keypoints1':torch.randn(1, 382, 2), 'scores1':torch.randn(1, 382), 'descriptors1':torch.randn(1, 256, 382)}
kpts0 = torch.randn(1, 382, 2)
scores0 = torch.randn(1, 382)
desc0 = torch.randn(1, 256, 382)
kpts1 = torch.randn(1, 382, 2)
scores1 = torch.randn(1, 382)
desc1 = torch.randn(1, 256, 382)
superglue = SuperGlue(config.get('superglue', {}))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# data = {'image':dummy_input}
pred0 = superglue(kpts0, scores0, desc0, kpts1, scores1, desc1)
print(pred0)
print('Converting to ONNX....')
torch.onnx.export(superglue,         # model being run 
    (kpts0, scores0, desc0, kpts1, scores1, desc1),       # model input (or a tuple for multiple inputs) 
    "./onnx/superglue_opset16.onnx",       # where to save the model  
    export_params=False,  # store the trained parameter weights inside the model file 
    opset_version=16,    # the ONNX version to export the model to 
    do_constant_folding=True,  # whether to execute constant folding for optimization 
    input_names = ['modelInput'],   # the model's input names 
    output_names = ['modelOutput']) # the model's output names
print(" ") 
print('Model has been converted to ONNX')



print('Converting to ONNX....')
torch.onnx.export(superglue,         # model being run 
    (kpts0, scores0, desc0, kpts1, scores1, desc1),       # model input (or a tuple for multiple inputs) 
    "./onnx/superglue_opset16_dynamic.onnx",       # where to save the model  
    export_params=False,  # store the trained parameter weights inside the model file 
    opset_version=16,    # the ONNX version to export the model to 
    do_constant_folding=True,  # whether to execute constant folding for optimization 
    input_names = ['modelInput'],   # the model's input names 
    output_names = ['modelOutput'], # the model's output names 
    dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                            'modelOutput' : {0 : 'batch_size'}}) 