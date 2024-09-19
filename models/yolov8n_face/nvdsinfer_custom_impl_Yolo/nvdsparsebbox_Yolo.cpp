/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>

#include "nvdsinfer_custom_impl.h"

// #include "utils.h"
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLIP(a, min, max) (MAX(MIN(a, max), min))

#define NMS_THRESH 0.45;

extern "C" bool
NvDsInferParseYoloFace(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferObjectDetectionInfo>& objectList);

static std::vector<NvDsInferObjectDetectionInfo>
nonMaximumSuppression(std::vector<NvDsInferObjectDetectionInfo> binfo)
{
  auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
    if (x1min > x2min) {
      std::swap(x1min, x2min);
      std::swap(x1max, x2max);
    }
    return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
  };

  auto computeIoU = [&overlap1D](NvDsInferObjectDetectionInfo& bbox1, NvDsInferObjectDetectionInfo& bbox2) -> float {
    float overlapX = overlap1D(bbox1.left, bbox1.left + bbox1.width, bbox2.left, bbox2.left + bbox2.width);
    float overlapY = overlap1D(bbox1.top, bbox1.top + bbox1.height, bbox2.top, bbox2.top + bbox2.height);
    float area1 = (bbox1.width) * (bbox1.height);
    float area2 = (bbox2.width) * (bbox2.height);
    float overlap2D = overlapX * overlapY;
    float u = area1 + area2 - overlap2D;
    return u == 0 ? 0 : overlap2D / u;
  };

  std::stable_sort(binfo.begin(), binfo.end(), [](const NvDsInferObjectDetectionInfo& b1, const NvDsInferObjectDetectionInfo& b2) {
    return b1.detectionConfidence > b2.detectionConfidence;
  });

  std::vector<NvDsInferObjectDetectionInfo> out;
  for (auto i : binfo) {
    bool keep = true;
    for (auto j : out) {
      if (keep) {
        float overlap = computeIoU(i, j);
        keep = overlap <= NMS_THRESH;
      }
      else {
        break;
      }
    }
    if (keep) {
      out.push_back(i);
    }
  }
  return out;
}

static std::vector<NvDsInferObjectDetectionInfo>
nmsAllClasses(std::vector<NvDsInferObjectDetectionInfo>& binfo)
{
  std::vector<NvDsInferObjectDetectionInfo> result = nonMaximumSuppression(binfo);
  return result;
}

// static void
// addFaceProposal(const float* landmarks, const uint& landmarksSizeRaw, const uint& netW, const uint& netH, const uint& b,
//     NvDsInferObjectDetectionInfo& bbi)
// {
//   uint landmarksSize = landmarksSizeRaw == 10 ? landmarksSizeRaw + 5 : landmarksSizeRaw;
//   bbi.mask = new float[landmarksSize];
//   for (uint p = 0; p < landmarksSize / 3; ++p) {
//     if (landmarksSizeRaw == 10) {
//       bbi.mask[p * 3 + 0] = clamp(landmarks[b * landmarksSizeRaw + p * 2 + 0], 0, netW);
//       bbi.mask[p * 3 + 1] = clamp(landmarks[b * landmarksSizeRaw + p * 2 + 1], 0, netH);
//       bbi.mask[p * 3 + 2] = 1.0;
//     }
//     else {
//       bbi.mask[p * 3 + 0] = clamp(landmarks[b * landmarksSize + p * 3 + 0], 0, netW);
//       bbi.mask[p * 3 + 1] = clamp(landmarks[b * landmarksSize + p * 3 + 1], 0, netH);
//       bbi.mask[p * 3 + 2] = landmarks[b * landmarksSize + p * 3 + 2];
//     }
//   }
//   bbi.mask_width = netW;
//   bbi.mask_height = netH;
//   bbi.mask_size = sizeof(float) * landmarksSize;
// }

static NvDsInferObjectDetectionInfo
convertBBox(const float& bx1, const float& by1, const float& bx2, const float& by2, const uint& netW, const uint& netH)
{
  NvDsInferObjectDetectionInfo b;

  float x1 = bx1;
  float y1 = by1;
  float x2 = bx2;
  float y2 = by2;

  x1 = CLIP(x1, 0, netW);
  y1 = CLIP(y1, 0, netH);
  x2 = CLIP(x2, 0, netW);
  y2 = CLIP(y2, 0, netH);

  b.left = x1;
  b.width = CLIP(x2 - x1, 0, netW);
  b.top = y1;
  b.height = CLIP(y2 - y1, 0, netH);



  return b;
}

static void
addBBoxProposal(const float bx1, const float by1, const float bx2, const float by2, const uint& netW, const uint& netH,
    const int maxIndex, const float maxProb, NvDsInferObjectDetectionInfo& bbi)
{
  bbi = convertBBox(bx1, by1, bx2, by2, netW, netH);

  if (bbi.width < 1 || bbi.height < 1) {
      return;
  }

  bbi.detectionConfidence = maxProb;
  bbi.classId = maxIndex;
}

static std::vector<NvDsInferObjectDetectionInfo>
decodeTensorYoloFace(const float* boxes, const float* scores, const float* landmarks, const uint& outputSize,
    const uint& landmarksSize, const uint& netW, const uint& netH, const std::vector<float>& preclusterThreshold)
{
  std::vector<NvDsInferObjectDetectionInfo> binfo;

  for (uint b = 0; b < outputSize; ++b) {
    float maxProb = scores[b];

    if (maxProb < preclusterThreshold[0]) {
      continue;
    }

    // float bxc = boxes[b * 4 + 0];
    // float byc = boxes[b * 4 + 1];
    // float bw = boxes[b * 4 + 2];
    // float bh = boxes[b * 4 + 3];

    // float bx1 = bxc - bw / 2;
    // float by1 = byc - bh / 2;
    // float bx2 = bx1 + bw;
    // float by2 = by1 + bh;

    float bx1 = boxes[b * 4 + 0];
    float by1 = boxes[b * 4 + 1];
    float bx2 = boxes[b * 4 + 2];
    float by2 = boxes[b * 4 + 3];

    NvDsInferObjectDetectionInfo bbi;

    addBBoxProposal(bx1, by1, bx2, by2, netW, netH, 0, maxProb, bbi);
    // std::cout<<landmarks[b * 15]<<" "<<landmarks[b * 15 + 1]<<" "<<landmarks[b * 15 + 2]<<std::endl;
    // addFaceProposal(landmarks, landmarksSize, netW, netH, b, bbi);
    for (unsigned int i2=0; i2 < 5; i2++) {
      bbi.landmark[i2*2] = lround(CLIP(landmarks[b * 15 + i2 * 3], 0, netW));
      bbi.landmark[i2*2 + 1] = lround(CLIP(landmarks[b * 15 + i2 * 3 + 1], 0, netH));
    }

    bbi.numLmks = 10;

    // std::cout<<bbi.top<<" "<<bbi.left<<" "<<bbi.width<<" "<<bbi.height<<" "<<bbi.detectionConfidence<<" "<<bbi.classId<<std::endl;
    binfo.push_back(bbi);
  }

  return binfo;
}

static bool NvDsInferParseCustomYoloFace(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                    NvDsInferNetworkInfo const &networkInfo,
                                    NvDsInferParseDetectionParams const &detectionParams,
                                    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo& boxes = outputLayersInfo[0];
  const NvDsInferLayerInfo& scores = outputLayersInfo[1];
  const NvDsInferLayerInfo& landmarks = outputLayersInfo[2];

  const uint outputSize = boxes.inferDims.d[0];
  const uint landmarksSize = landmarks.inferDims.d[1];

  std::vector<NvDsInferObjectDetectionInfo> objects = decodeTensorYoloFace((const float*) (boxes.buffer),
      (const float*) (scores.buffer), (const float*) (landmarks.buffer), outputSize, landmarksSize, networkInfo.width,
      networkInfo.height, detectionParams.perClassPreclusterThreshold);
  // std::cout<<"objects size: "<<objects.size()<<std::endl;
  objectList.clear();
  objectList = nmsAllClasses(objects);
  // std::cout<<"objects size after nms: "<<objectList.size()<<std::endl;
  // for (auto r : objectList) {
  //   std::cout << "bbox: " << r.left << " " << r.top << " " << r.width << " " << r.height << " " << r.detectionConfidence << " " << r.classId << std::endl;
  // }

  return true;
}

extern "C" bool
NvDsInferParseYoloFace(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  return NvDsInferParseCustomYoloFace(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloFace);