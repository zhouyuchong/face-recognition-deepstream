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

extern "C" bool NvDsInferParseCustomRetinaFace(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                               NvDsInferNetworkInfo const &networkInfo,
                                               NvDsInferParseDetectionParams const &detectionParams,
                                               std::vector<NvDsInferObjectDetectionInfo> &objectList);
struct Bbox {
    int x1, y1, x2, y2;
    float score;
};
struct anchorBox {
    float cx;
    float cy;
    float sx;
    float sy;
};
void postprocessing(float *bbox, float *conf, float bbox_threshold, float nms_threshold, unsigned int topk, int width,
                    int height, std::vector<NvDsInferObjectDetectionInfo> &objectList);
void create_anchor_retinaface(std::vector<anchorBox> &anchor, int w, int h);
bool cmp(NvDsInferObjectDetectionInfo a, NvDsInferObjectDetectionInfo b);
void nms(std::vector<NvDsInferObjectDetectionInfo> &input_boxes, float NMS_THRESH);

void postprocessing(float *bbox, float *conf, float bbox_threshold, float nms_threshold, unsigned int topk, int width,
                    int height, std::vector<NvDsInferObjectDetectionInfo> &objectList) {
    std::vector<anchorBox> anchor;
    create_anchor_retinaface(anchor, width, height);

    for (unsigned int i = 0; i < anchor.size(); ++i) {
        if (*(conf + 1) > bbox_threshold) {
            anchorBox tmp = anchor[i];
            anchorBox tmp1;
            NvDsInferObjectDetectionInfo result;
            result.classId = 0;

            // decode bbox
            tmp1.cx = tmp.cx + *bbox * 0.1 * tmp.sx;
            tmp1.cy = tmp.cy + *(bbox + 1) * 0.1 * tmp.sy;
            tmp1.sx = tmp.sx * exp(*(bbox + 2) * 0.2);
            tmp1.sy = tmp.sy * exp(*(bbox + 3) * 0.2);

            result.left = (tmp1.cx - tmp1.sx / 2) * width;
            result.top = (tmp1.cy - tmp1.sy / 2) * height;
            result.width = (tmp1.cx + tmp1.sx / 2) * width - result.left;
            result.height = (tmp1.cy + tmp1.sy / 2) * height - result.top;

            // Clip object box coordinates to network resolution
            result.left = CLIP(result.left, 0, width - 1);
            result.top = CLIP(result.top, 0, height - 1);
            result.width = CLIP(result.width, 0, width - 1);
            result.height = CLIP(result.height, 0, height - 1);

            result.detectionConfidence = *(conf + 1);
            objectList.push_back(result);
        }
        bbox += 4;
        conf += 2;
    }
    std::sort(objectList.begin(), objectList.end(), cmp);
    nms(objectList, nms_threshold);
    if (objectList.size() > topk)
        objectList.resize(topk);
}

void create_anchor_retinaface(std::vector<anchorBox> &anchor, int w, int h) {
    anchor.clear();
    std::vector<std::vector<int>> feature_map(3), min_sizes(3);
    float steps[] = {8, 16, 32};
    for (unsigned int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h / steps[i]));
        feature_map[i].push_back(ceil(w / steps[i]));
    }
    std::vector<int> minsize1 = {10, 20};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {32, 64};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {128, 256};
    min_sizes[2] = minsize3;

    for (unsigned int k = 0; k < feature_map.size(); ++k) {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i) {
            for (int j = 0; j < feature_map[k][1]; ++j) {
                for (unsigned int l = 0; l < min_size.size(); ++l) {
                    float s_kx = min_size[l] * 1.0 / w;
                    float s_ky = min_size[l] * 1.0 / h;
                    float cx = (j + 0.5) * steps[k] / w;
                    float cy = (i + 0.5) * steps[k] / h;
                    anchorBox axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                }
            }
        }
    }
}

bool cmp(NvDsInferObjectDetectionInfo a, NvDsInferObjectDetectionInfo b) {
    if (a.detectionConfidence > b.detectionConfidence)
        return true;
    return false;
}

void nms(std::vector<NvDsInferObjectDetectionInfo> &input_boxes, float NMS_THRESH) {
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).width + 1) * (input_boxes.at(i).height + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = std::max(input_boxes[i].left, input_boxes[j].left);
            float yy1 = std::max(input_boxes[i].top, input_boxes[j].top);
            float xx2 =
                std::min(input_boxes[i].left + input_boxes[i].width, input_boxes[j].left + input_boxes[j].width);
            float yy2 =
                std::min(input_boxes[i].top + input_boxes[i].height, input_boxes[j].top + input_boxes[j].height);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}
bool NvDsInferParseCustomRetinaFace(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                    NvDsInferNetworkInfo const &networkInfo,
                                    NvDsInferParseDetectionParams const &detectionParams,
                                    std::vector<NvDsInferObjectDetectionInfo> &objectList) {
    // Get output indexes
    static int bboxLayerIndex = -1;
    static int confLayerIndex = -1;
    static int lmkLayerIndex = -1;
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
        if (strcmp(outputLayersInfo[i].layerName, "bbox") == 0) {
            bboxLayerIndex = i;
        } else if (strcmp(outputLayersInfo[i].layerName, "conf") == 0) {
            confLayerIndex = i;
        } else if (strcmp(outputLayersInfo[i].layerName, "lmk") == 0) {
            lmkLayerIndex = i;
        }
    }
    if ((bboxLayerIndex == -1) || (confLayerIndex == -1) || (lmkLayerIndex == -1)) {
        std::cerr << "Could not find output layer buffer while parsing" << std::endl;
        return false;
    }

    // Host memory for "decode"
    float *bbox = (float *)outputLayersInfo[bboxLayerIndex].buffer;
    float *conf = (float *)outputLayersInfo[confLayerIndex].buffer;
    float *lmk = (float *)outputLayersInfo[lmkLayerIndex].buffer;

    // Get thresholds and topk value
    // const float bbox_threshold = detectionParams.perClassPreclusterThreshold[0];
    // const float nms_threshold = detectionParams.perClassPostclusterThreshold[0];
    // const unsigned int topk = detectionParams.numClassesConfigured;
    const float bbox_threshold = 0.5;
    const float nms_threshold = 0.5;
    const unsigned int topk = 100;

    // Do post processing
    postprocessing(bbox, conf, bbox_threshold, nms_threshold, topk, networkInfo.width, networkInfo.height, objectList);
    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomRetinaFace);