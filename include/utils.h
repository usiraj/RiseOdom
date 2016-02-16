/*
 * utils.h
 *
 *  Created on: Feb 6, 2015
 *      Author: usama
 */

#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include "transformation_matrix.h"
struct IdxCorrespondance{
	int Idx1;
	int Idx2;
};
typedef std::vector<IdxCorrespondance> ListIdxCorrespondance;
typedef std::vector<cv::KeyPoint> KPtsList;
typedef std::vector<cv::DMatch> MatchList;
typedef std::vector<MatchList> ListMatchList;
typedef void(*MotionFilterFunc)(const ListIdxCorrespondance &in, const cv::Mat &_pts3DA, cv::Mat &_pts2DB, const cv::Matx33f &KMat,
								ListIdxCorrespondance &out, const int _param1, const float param2, cv::Mat *rvec, cv::Mat *tvec);
void print_2dmats(const cv::Mat &mat);
void print_2dmat(const cv::Mat &mat);
void print_cloud2d(const cv::Mat &mat);
void print_cloud3d(const cv::Mat &mat);
void point_cloud2d(const KPtsList&, cv::Mat &outpts2d);
void point_cloud2d_64fc1(const KPtsList &kpts, cv::Mat &outpts2d);
void point_cloud3d(const KPtsList&, const cv::Mat &pts3d, cv::Mat &outpts3d, cv::Mat *outpts2d=0);
cv::Mat blend_disparity(const cv::Mat &imga, const cv::Mat &udisp, const cv::Mat *_mask = 0);
cv::Mat blend_image(const cv::Mat &imga, const cv::Mat &imgb);
cv::Mat draw_lines(const cv::Mat &imga, const KPtsList &kpl1, const KPtsList &kpl2, const cv::Scalar &clr=cv::Scalar(0, 255, 0));
cv::Mat draw_allkeypts(const cv::Mat &imga, const KPtsList &kpl1, const KPtsList &kpl2, const ListIdxCorrespondance &list);
cv::Mat draw_matched_keypts(const cv::Mat &imga, const KPtsList &kpl1, const KPtsList &kpl2, const std::vector<int> *inliers=0);
cv::Mat draw_matched_keypts(const cv::Mat &imga, const KPtsList& kpl1, const KPtsList& kpl2, const int clr);
void correspondance_list(const MatchList &matches, ListIdxCorrespondance &cr, const bool _flipassociation=false);
void filtercorrespondance(const KPtsList &kpl1, const KPtsList &kpl2,
						  const ListIdxCorrespondance &cr, ListIdxCorrespondance &crout, const float sigma=1.0);
void filtercorrespondanceransac(const ListIdxCorrespondance &cr,const cv::Mat &_pts3d,const cv::Mat &_pts2d,
								const cv::Matx33f &cam,ListIdxCorrespondance &crout,const int iterations=80, const float reproj=8.0,
								cv::Mat *rvec=NULL, cv::Mat *tvec=NULL);
void select_descriptorkeypts(const std::vector<int> &_indices, const KPtsList &kpl1, const cv::Mat &des1,
	KPtsList &kplout, cv::Mat &desout);
void corresponding_descriptors(const ListIdxCorrespondance &cr, const cv::Mat &des1,
							   const cv::Mat &des2, cv::Mat &des1_m, cv::Mat &des2_m);
void corresponding_rowmat(const ListIdxCorrespondance &cr, const cv::Mat &mat1,
						  const cv::Mat &mat2, cv::Mat &out1, cv::Mat &out2);
void corresponding_keypts(const ListIdxCorrespondance &cr, const KPtsList &kp1, const KPtsList &kp2,
						  KPtsList &kp1_m, KPtsList &kp2_m);
void ratiotest_lowe(const ListMatchList &matches, MatchList &good,const float ratio=0.45);
void match_across(const cv::Mat &des1, const cv::Mat &des2, ListIdxCorrespondance &list,
				  const float ratio, cv::Ptr<cv::DescriptorMatcher> matcher);
void match_acrossflannl2(const cv::Mat &des, const cv::Mat &des2, ListIdxCorrespondance &list,
						 const float ratio=0.45);
void match_acrossflannhamming(const cv::Mat &des, const cv::Mat &des2, ListIdxCorrespondance &list,
							  const float ratio=0.45);
void match_acrossbfl2(const cv::Mat &des, const cv::Mat &des2, ListIdxCorrespondance &list,
					  const float ratio=0.45, const int norm=cv::NORM_L2);
// some utility functions for ListIdxCorrespondance
void correspondance_decompose(const ListIdxCorrespondance &list, std::vector<int> &out1, std::vector<int> &out2);
void select_rowmat(const std::vector<int> &rows, const cv::Mat &inmat, cv::Mat &outmat);
bool inIntList(const int _num, const std::vector<int> &_list);
void select_complement_rowmat(const std::vector<int> &rows, const cv::Mat &inmat, cv::Mat &outmat);
#endif
