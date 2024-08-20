/*
 * @Author: zhouyuchong
 * @Date: 2024-02-26 14:51:58
 * @Description: 
 * @LastEditors: zhouyuchong
 * @LastEditTime: 2024-08-16 11:38:39
 */
#ifndef _DAMONZZZ_ALIGNER_H_
#define _DAMONZZZ_ALIGNER_H_

#include "opencv2/opencv.hpp"


namespace alignnamespace {
class Aligner {
public:
	Aligner();
	~Aligner();

	cv::Mat Align(const cv::Mat & dst, int model_type);
	bool validLmks(float landmarks[10]);
	
private:
	class Impl;
	Impl* impl_;
};

} // namespace alignnamespace

#endif // !_DAMONZZZ_ALIGNER_H_