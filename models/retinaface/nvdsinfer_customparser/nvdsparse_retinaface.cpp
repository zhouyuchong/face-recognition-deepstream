/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLIP(a, min, max) (MAX(MIN(a, max), min))
#define CONF_THRESH 0.1
#define VIS_THRESH 0.75
#define NMS_THRESH 0.4

extern "C" bool NvDsInferParseCustomRetinaface(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList);

static constexpr int LOCATIONS = 4;
static constexpr int ANCHORS = 10;

struct alignas(float) Detection{
    float bbox[LOCATIONS];
    float score;
    float anchor[ANCHORS];
};

void create_anchor_retinaface(std::vector<Detection>& res, float *output, float conf_thresh, int width, int height) {
    int det_size = sizeof(Detection) / sizeof(float);
    for (int i = 0; i < output[0]; i++){
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        
        Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        det.bbox[0] = CLIP(det.bbox[0], 0, width - 1);
        det.bbox[1] = CLIP(det.bbox[1] , 0, height -1);
        det.bbox[2] = CLIP(det.bbox[2], 0, width - 1);
        det.bbox[3] = CLIP(det.bbox[3], 0, height - 1);
        
        res.push_back(det);
        
    }
}

bool cmp(Detection& a, Detection& b) {
    return a.score > b.score;
}


float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

void nms_and_adapt(std::vector<Detection>& det, std::vector<Detection>& res, float nms_thresh, int width, int height) {
    std::sort(det.begin(), det.end(), cmp);
    for (unsigned int m = 0; m < det.size(); ++m) {
        auto& item = det[m];
        res.push_back(item);
        for (unsigned int n = m + 1; n < det.size(); ++n) {
            if (iou(item.bbox, det[n].bbox) > nms_thresh) {
                det.erase(det.begin()+n);
                --n;
            }
        }
    }
    // crop larger area for better alignment performance
    // there I choose to crop 50 more pixel
    for (unsigned int m = 0; m < res.size(); ++m) {
        res[m].bbox[0] = CLIP(res[m].bbox[0] - 10, 0, width - 1);
        res[m].bbox[1] = CLIP(res[m].bbox[1] - 10, 0, height -1);
        res[m].bbox[2] = CLIP(res[m].bbox[2] + 20, 0, width - 1);
        res[m].bbox[3] = CLIP(res[m].bbox[3] + 20, 0, height - 1);
    }
}


static bool NvDsInferParseRetinaface(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                    NvDsInferNetworkInfo const &networkInfo,
                                    NvDsInferParseDetectionParams const &detectionParams,
                                    std::vector<NvDsInferObjectDetectionInfo> &objectList) {
    
  
    float *output = (float*)(outputLayersInfo[0].buffer);
    std::vector<Detection> temp;
    std::vector<Detection> res;
    create_anchor_retinaface(temp, output, CONF_THRESH, networkInfo.width, networkInfo.height);
    nms_and_adapt(temp, res, NMS_THRESH, networkInfo.width, networkInfo.height);

    for(auto& r : res) {
        
        if(r.score<=VIS_THRESH) continue;

	    NvDsInferParseObjectInfo oinfo;  
	    oinfo.classId = 0;
	    oinfo.left    = static_cast<unsigned int>(r.bbox[0]);
	    oinfo.top     = static_cast<unsigned int>(r.bbox[1]);
	    oinfo.width   = static_cast<unsigned int>(r.bbox[2]-r.bbox[0]);
	    oinfo.height  = static_cast<unsigned int>(r.bbox[3]-r.bbox[1]);
	    oinfo.detectionConfidence = r.score;
        objectList.push_back(oinfo);
             
    }
    return true;
}


extern "C" bool NvDsInferParseCustomRetinaface(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    return NvDsInferParseRetinaface(
        outputLayersInfo, networkInfo, detectionParams, objectList);
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomRetinaface);
