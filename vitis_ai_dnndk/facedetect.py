'''
Copyright 2020 Avnet Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from ctypes import *
import cv2
import os
import threading
import time
import sys
import numpy as np
from numpy import float32

from vitis_ai_dnndk import n2cube
from vitis_ai_dnndk import dputils 

KERNEL = "densebox"
NODE_INPUT = "L0"
NODE_CONV = "pixel_conv"
NODE_OUTPUT = "bb_output"

def time_it(msg,start,end):
    print("[INFO] {} took {:.8} seconds".format(msg,end-start))

def nms_boxes(boxes, scores, nms_threshold):
    """
    Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.
        nms_threshold: threshold for NMS algorithm

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2-x1+1)*(y2-y1+1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 1)
        h1 = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_threshold)[0]  # threshold
        order = order[inds + 1]

    return keep


class FaceDetect():
  def __init__(self, detThreshold=0.55, nmsThreshold=0.35):
    self.detThreshold = detThreshold
    self.nmsThreshold = nmsThreshold
    self.dpuKernel = []
    self.dpuTask = []

  def start(self):
    """Attach to DPU driver and prepare for runing"""
    #n2cube.dpuOpen()

    """Create DPU Kernels for DenseBox"""
    self.dpuKernel = n2cube.dpuLoadKernel(KERNEL)

    """Create DPU Tasks from DPU Kernel"""
    self.dpuTask = n2cube.dpuCreateTask(self.dpuKernel, 0)


  def process(self,img):
    kernel = self.dpuKernel
    task = self.dpuTask

    imgHeight = img.shape[0]
    imgWidth  = img.shape[1]

    conv_in_tensor = n2cube.dpuGetInputTensor(task, NODE_INPUT)
    imHeight = n2cube.dpuGetTensorHeight(conv_in_tensor)
    imWidth = n2cube.dpuGetTensorWidth(conv_in_tensor)
    scale_h = imgHeight / imHeight
    scale_w = imgWidth / imWidth

    """Load image to DPU"""
    dputils.dpuSetInputImage2(task, NODE_INPUT, img)
        
    """Model run on DPU"""
    n2cube.dpuRunTask(task)

    conv_out_tensor = n2cube.dpuGetOutputTensor(task, NODE_OUTPUT)
    tensorSize = n2cube.dpuGetTensorSize(conv_out_tensor)
    outHeight = n2cube.dpuGetTensorHeight(conv_out_tensor)
    outWidth = n2cube.dpuGetTensorWidth(conv_out_tensor)
	
    outAddr = n2cube.dpuGetOutputTensorAddress(task, NODE_CONV)
    size = n2cube.dpuGetOutputTensorSize(task, NODE_CONV)
    channel = n2cube.dpuGetOutputTensorChannel(task, NODE_CONV)
    out_scale = n2cube.dpuGetOutputTensorScale(task, NODE_CONV)
    softmax = np.zeros(size,dtype=float32)
        
    """ Output data format convert """
    bb = n2cube.dpuGetOutputTensorInHWCFP32(task, NODE_OUTPUT, tensorSize)
    bboxes = np.reshape( bb, (-1, 4) )
	#

    """ Get original face boxes """
    gy = np.arange(0,outHeight)
    gx = np.arange(0,outWidth)
    [x,y] = np.meshgrid(gx,gy)
    x = x.ravel()*4
    y = y.ravel()*4
    bboxes[:,0] = bboxes[:,0] + x
    bboxes[:,1] = bboxes[:,1] + y
    bboxes[:,2] = bboxes[:,2] + x
    bboxes[:,3] = bboxes[:,3] + y

    """ Run softmax """
    softmax = n2cube.dpuRunSoftmax(outAddr, channel, size // channel, out_scale)

    """ Only keep faces for which prob is above detection threshold """
    scores = np.reshape( softmax, (-1, 2))
    prob = scores[:,1] 
    keep_idx = prob.ravel() > self.detThreshold
    bboxes = bboxes[ keep_idx, : ]
    bboxes = np.array( bboxes, dtype=np.float32 )
    prob = prob[ keep_idx ]
	
    """ Perform Non-Maxima Suppression """
    face_indices = []
    if ( len(bboxes) > 0 ):
        face_indices = nms_boxes( bboxes, prob, self.nmsThreshold );

    faces = bboxes[face_indices]

    # extract bounding box for each face
    for i, face in enumerate(faces):
        xmin = max(face[0] * scale_w, 0 )
        ymin = max(face[1] * scale_h, 0 )
        xmax = min(face[2] * scale_w, imgWidth )
        ymax = min(face[3] * scale_h, imgHeight )
        faces[i] = ( int(xmin),int(ymin),int(xmax),int(ymax) )

    return faces

  def stop(self):
    """Destroy DPU Tasks & free resources"""
    n2cube.dpuDestroyTask(self.dpuTask)

    """Destroy DPU Tasks & free resources"""
    rtn = n2cube.dpuDestroyKernel(self.dpuKernel)

    """Dettach from DPU driver & release resources"""
    #n2cube.dpuClose()

    self.dpuTask = []
    self.dpuKernel = []

