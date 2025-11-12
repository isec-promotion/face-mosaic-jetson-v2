/*
 * YOLOv8 Custom Bounding Box Parser for DeepStream
 * 
 * This parser handles YOLOv8 output format:
 * - Single output tensor: output0
 * - Shape: [1, 5, 8400] for face detection (num_classes=1 + 4 bbox coords)
 * - Format: [batch, num_classes + 4, num_anchors]
 */

#include <algorithm>
#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))

/* YOLOv8 output format:
 * output0: [1, 5, 8400]
 *   - 5 = 1 class score + 4 bbox coordinates (x, y, w, h)
 *   - 8400 = number of detection anchors
 */

extern "C" bool NvDsInferParseYoloV8(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    if (outputLayersInfo.empty() || outputLayersInfo.size() != 1) {
        std::cerr << "ERROR: YOLOv8 expects exactly 1 output layer, got " 
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    const NvDsInferLayerInfo &output = outputLayersInfo[0];
    
    // YOLOv8 output shape can be:
    // 2D: [num_classes+4, num_anchors] = [5, 8400]
    // 3D: [batch, num_classes+4, num_anchors] = [1, 5, 8400]
    
    int batch = 1;
    int num_data = 0;
    int num_anchors = 0;
    
    if (output.inferDims.numDims == 2) {
        // 2D format: [num_classes+4, num_anchors]
        num_data = output.inferDims.d[0];
        num_anchors = output.inferDims.d[1];
    } else if (output.inferDims.numDims == 3) {
        // 3D format: [batch, num_classes+4, num_anchors]
        batch = output.inferDims.d[0];
        num_data = output.inferDims.d[1];
        num_anchors = output.inferDims.d[2];
    } else {
        std::cerr << "ERROR: YOLOv8 output should have 2 or 3 dimensions, got " 
                  << output.inferDims.numDims << std::endl;
        return false;
    }
    
    const int num_classes = num_data - 4;  // Should be 1 for face detection
    
    std::cout << "INFO: YOLOv8 output shape: [" << batch << ", " << num_data 
              << ", " << num_anchors << "]" << std::endl;
    std::cout << "INFO: Detected " << num_classes << " classes" << std::endl;

    const float* data = (const float*)output.buffer;
    
    // Parse detections
    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
        // Get class scores (only 1 class for face detection)
        float max_score = 0.0f;
        int max_class_id = 0;
        
        for (int cls = 0; cls < num_classes; cls++) {
            // Index calculation for 2D or 3D tensor
            int score_idx = (4 + cls) * num_anchors + anchor_idx;
            float score = data[score_idx];
            
            if (score > max_score) {
                max_score = score;
                max_class_id = cls;
            }
        }
        
        // Apply threshold
        if (max_score < detectionParams.perClassThreshold[max_class_id]) {
            continue;
        }
        
        // Get bounding box coordinates
        // YOLOv8 format: [x_center, y_center, width, height] in normalized coords
        int x_idx = 0 * num_anchors + anchor_idx;
        int y_idx = 1 * num_anchors + anchor_idx;
        int w_idx = 2 * num_anchors + anchor_idx;
        int h_idx = 3 * num_anchors + anchor_idx;
        
        float x_center = data[x_idx];
        float y_center = data[y_idx];
        float width = data[w_idx];
        float height = data[h_idx];
        
        // Convert to top-left corner coordinates
        float x1 = x_center - width / 2.0f;
        float y1 = y_center - height / 2.0f;
        
        // Scale to network input dimensions
        x1 *= networkInfo.width;
        y1 *= networkInfo.height;
        width *= networkInfo.width;
        height *= networkInfo.height;
        
        // Clip to valid range
        x1 = CLIP(x1, 0.0f, (float)networkInfo.width - 1);
        y1 = CLIP(y1, 0.0f, (float)networkInfo.height - 1);
        width = CLIP(width, 0.0f, (float)networkInfo.width);
        height = CLIP(height, 0.0f, (float)networkInfo.height);
        
        // Add to object list
        NvDsInferParseObjectInfo obj;
        obj.classId = max_class_id;
        obj.left = x1;
        obj.top = y1;
        obj.width = width;
        obj.height = height;
        obj.detectionConfidence = max_score;
        
        objectList.push_back(obj);
    }
    
    std::cout << "INFO: Detected " << objectList.size() << " objects before NMS" << std::endl;
    
    return true;
}

/* C linkage */
extern "C" bool NvDsInferParseCustomYoloV8(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    return NvDsInferParseYoloV8(outputLayersInfo, networkInfo, detectionParams, objectList);
}
