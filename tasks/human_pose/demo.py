import cv2
import PIL.Image

import json
import trt_pose.coco
#import trt_pose.models

import torch
import torch2trt
from torch2trt import TRTModule

import time

import torchvision.transforms as transforms

from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

def execute(image):
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    draw_objects(image, counts, objects, peaks)
    #print(image.mode)
    #print(image.size)
    #image_w.value = bgr8_to_jpeg(image[:, ::-1, :])
    #print(image)
    #result_image = bytes(cv2.imencode('.jpg',image)[1])
    #print(result_image)
    return image
    

    
def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.resize(image, (224,224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

# TesorRT
#num_parts = len(human_pose['keypoints'])
#num_links = len(human_pose['skeleton'])

t0 = time.time()
print('model load start')
#model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
#MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
#model.load_state_dict(torch.load(MODEL_WEIGHTS))


#WIDTH = 224
#HEIGHT = 224
#data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

#model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
#torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
t1 = time.time()
print('model load complete ',t1-t0)

'''
t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()
print(50.0 / (t1 - t0))
'''

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')


parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)


#camera = USBCamera(width=WIDTH, height=HEIGHT, capture_fps=30)
#camera.running = True

#if cap.isOpen():
#    print('width: {}, height: {}'.format(cap.get(3), cap.get(4)))

#fcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('/home/interminds/result.mp4', fcc, 20, (1280, 720))

#t2 = time.time()
#cap = cv2.VideoCapture('/home/interminds/Videos/XND-6010_20210324101510.avi')
#cap = cv2.VideoCapture('/home/interminds/Videos/XND-6010_20210420150717.mp4')
cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while(cap.isOpened()):
    ret, frame = cap.read()
    #ret, frame = camera.read()
    #t3 = time.time()
    
    if ret:
        #frame = cv2.resize(frame, (224, 224))
        frame = execute(frame)
        cv2.imshow('video',frame)
        k = cv2.waitKey(1) & 0xFF
        #if (t3-t2) > 5:
        if k == 27:
            break

    else:
        print('error')
        break

cap.release()
#out.release()
cv2.destroyAllWindows()
