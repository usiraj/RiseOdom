/*
 * utils.cpp
 *
 *  Created on: Feb 6, 2015
 *      Author: usama
 */

#include "utils.h"
#include <stdio.h>

void print_2dmats(const cv::Mat &mat){
    float _val;
	short _vals;
	unsigned short _valu;
	for ( int _r = 0; _r < mat.rows; ++_r){
		printf("\n");
		for ( int _c = 0; _c < mat.cols; ++_c){
			if ( mat.type() == CV_64F ){
				_val = mat.at<double>(_r, _c);
				printf("%.8le\t", _val);
			} else if ( mat.type() == CV_32F ){
				_val = mat.at<float>(_r, _c);
				printf("%e\t", _val);
			} else if (mat.type() == CV_16U ){
				_valu = mat.at<unsigned short>(_r, _c);
				printf("%d\t", _valu);
			} else if (mat.type() == CV_16S ){
				_vals = mat.at<short>(_r, _c);
				printf("%d\t", _vals);
			}
		}
		fflush(stdout);
	}
	printf("\n");
}
void print_2dmat(const cv::Mat &mat){
    float _val;
	short _vals;
	unsigned short _valu;
	for ( int _r = 0; _r < mat.rows; ++_r){
		printf("\n");
		for ( int _c = 0; _c < mat.cols; ++_c){
			if ( mat.type() == CV_64F ){
				_val = mat.at<double>(_r, _c);
				printf("%lf\t", _val);
			} else if ( mat.type() == CV_32F ){
				_val = mat.at<float>(_r, _c);
				printf("%f\t", _val);
			} else if (mat.type() == CV_16U ){
				_valu = mat.at<unsigned short>(_r, _c);
				printf("%d\t", _valu);
			} else if (mat.type() == CV_16S ){
				_vals = mat.at<short>(_r, _c);
				printf("%d\t", _vals);
			}
		}
		fflush(stdout);
	}
	printf("\n");
}
void print_cloud2d(const cv::Mat& mat)
{
	if ( mat.depth() == CV_32F ){
		for ( cv::MatConstIterator_<cv::Point2f> itr = mat.begin<cv::Point2f>() ; itr < mat.end<cv::Point2f>(); ++itr ){
			printf("%f, %f\n", (*itr).x, (*itr).y );
		}
	} else if (mat.depth() == CV_64F) {
		for (cv::MatConstIterator_<cv::Point2d> itr = mat.begin<cv::Point2d>(); itr < mat.end<cv::Point2d>(); ++itr){
			printf("%lf, %lf\n", (*itr).x, (*itr).y );
		}
	} else {
		printf("Matrix Not of floating type");
	}
	fflush(stdout);
}
void print_cloud3d(const cv::Mat& mat)
{
	if ( mat.depth() == CV_32F ){
		for ( cv::MatConstIterator_<cv::Point3f> itr = mat.begin<cv::Point3f>() ; itr < mat.end<cv::Point3f>(); ++itr ){
			printf("%f, %f, %f\n", (*itr).x, (*itr).y , (*itr).z);
		}
	} else if (mat.depth() == CV_64F) {
		for (cv::MatConstIterator_<cv::Point3d> itr = mat.begin<cv::Point3d>(); itr < mat.end<cv::Point3d>(); ++itr){
			printf("%lf, %lf, %lf\n", (*itr).x, (*itr).y , (*itr).z);
		}
	} else {
		printf("Matrix Not of floating type");
	}
	fflush(stdout);
}
cv::Mat draw_lines(const cv::Mat& imga, const KPtsList& kpl1, const KPtsList& kpl2, const cv::Scalar& clr)
{
	cv::Mat outimage;
	if (imga.channels() == 1){
		cv::Mat arr[3];
		arr[0] = imga; arr[1] = imga; arr[2] = imga;
		cv::merge(arr, 3, outimage);
	} else {
		outimage = imga.clone();
	}
	if ( kpl1.size() != kpl2.size() ){
		return outimage;
	}
	int _n = kpl1.size();
	for ( int _i = 0; _i < _n; ++_i ){
		cv::line(outimage, kpl1[_i].pt, kpl2[_i].pt, clr, 1);
	}
	return outimage;
}
cv::Mat draw_matched_keypts(const cv::Mat &imga, const KPtsList& kpl1, const KPtsList& kpl2, const int clr){
	cv::Mat outimage;
	cv::Scalar colour;
	if (imga.channels() == 1){
		cv::Mat arr[3];
		arr[0] = imga; arr[1] = imga; arr[2] = imga;
		cv::merge(arr, 3, outimage);
	} else {
		outimage = imga.clone();
	}
	switch (clr ){
		case 0:
			colour = cv::Scalar(10, 100, 10);
			break;
		case 1:
			colour = cv::Scalar(10, 200, 10);
			break;
		case 2:
			colour = cv::Scalar(10, 200, 100);
			break;
		case 3:
			colour = cv::Scalar(100, 200, 10);
			break;
		case 4:
			colour = cv::Scalar(10, 200, 200);
			break;
		case 5:
			colour = cv::Scalar(200, 200, 10);
			break;
		case 6:
			colour = cv::Scalar(255, 255, 10);
			break;
		default:
			colour = cv::Scalar(10, 255, 255);
			break;
	}
	cv::drawKeypoints(outimage, kpl1, outimage, cv::Scalar(120, 10, 10));
	cv::drawKeypoints(outimage, kpl2, outimage, cv::Scalar(10, 10, 120));
	outimage = draw_lines(outimage, kpl1, kpl2, colour);
	return outimage;
}
cv::Mat draw_matched_keypts(const cv::Mat &imga, const KPtsList& kpl1, const KPtsList& kpl2, const std::vector< int >* inliers)
{
	cv::Mat outimage;
	if (imga.channels() == 1){
		cv::Mat arr[3];
		arr[0] = imga; arr[1] = imga; arr[2] = imga;
		cv::merge(arr, 3, outimage);
	} else {
		outimage = imga.clone();
	}	
	cv::drawKeypoints(outimage, kpl1, outimage, cv::Scalar(120, 10, 10));
	cv::drawKeypoints(outimage, kpl2, outimage, cv::Scalar(10, 10, 120));
	outimage = draw_lines(outimage, kpl1, kpl2, cv::Scalar(10, 120, 10));
	if (inliers != NULL ){
		KPtsList _kpl1, _kpl2;
		for ( std::vector<int>::const_iterator itr = inliers->begin(); itr < inliers->end(); ++itr ){
			_kpl1.push_back(kpl1[*itr]);
			_kpl2.push_back(kpl2[*itr]);
		}
		cv::drawKeypoints(outimage, _kpl1, outimage, cv::Scalar(250, 30, 30));
		cv::drawKeypoints(outimage, _kpl2, outimage, cv::Scalar(30, 30, 250));
		outimage = draw_lines(outimage, _kpl1, _kpl2, cv::Scalar(30, 250, 30));
	}	
	return outimage;
}
cv::Mat draw_allkeypts(const cv::Mat& imga, const KPtsList& kpl1, const KPtsList& kpl2, const ListIdxCorrespondance& list)
{
	cv::Mat outimage;
	if (imga.channels() == 1){
		cv::Mat arr[3];
		arr[0] = imga; arr[1] = imga; arr[2] = imga;
		cv::merge(arr, 3, outimage);
	} else {
		outimage = imga.clone();
	}
	cv::drawKeypoints(outimage, kpl1, outimage, cv::Scalar(120, 10, 10));
	cv::drawKeypoints(outimage, kpl2, outimage, cv::Scalar(10, 10, 120));
	KPtsList kpl1m, kpl2m;
	corresponding_keypts(list, kpl1, kpl2, kpl1m, kpl2m);
	cv::drawKeypoints(outimage, kpl1m, outimage, cv::Scalar(220, 60, 10));
	cv::drawKeypoints(outimage, kpl2m, outimage, cv::Scalar(10, 60, 220));
	outimage = draw_lines(outimage, kpl1m, kpl2m, cv::Scalar(10, 240, 10));
	return outimage;
}
cv::Mat blend_image(const cv::Mat& imga, const cv::Mat& imgb)
{
	cv::Mat outimage;
	cv::addWeighted(imga, 0.5, imgb , 0.5, 0.0, outimage);
	return outimage;	
}
cv::Mat blend_disparity(const cv::Mat& imga, const cv::Mat& udisp, const cv::Mat* _mask)
{
	cv::Mat outimage, zero, clr, dimg;
	if (imga.channels() == 1){
		cv::Mat arr[3];
		arr[0] = imga; arr[1] = imga; arr[2] = imga;
		cv::merge(arr, 3, outimage);
	} else {
		outimage = imga.clone();
	}
	cv::Mat arr[3];
	zero = cv::Mat::zeros(udisp.rows, udisp.cols, udisp.type());
	if ( _mask != NULL ){
		cv::bitwise_and(udisp, *_mask, clr);
	} else {
		clr = udisp.clone();
	}
	arr[0] = clr; arr[2] = clr; arr[1] = zero;
	cv::merge(arr, 3, dimg);
	cv::addWeighted(outimage, 0.5, dimg , 0.5, 0.0, outimage);
	return outimage;
}
void filtercorrespondance(const KPtsList& kpl1, const KPtsList& kpl2, const ListIdxCorrespondance& cr,
						  ListIdxCorrespondance& crout, const float sigma)
{
	crout.clear();
	int _n = cr.size();
	if (_n == 0) return;
	float _val, _dx, _dy;
	int _idx1, _idx2;
	cv::Mat _dt = cv::Mat(_n, 1, CV_64FC1);	
	// calculate distances
	cv::KeyPoint kp;
	kp.pt;
	for ( int _i = 0; _i < _n ; ++_i ){
		_idx1 = cr[_i].Idx1; _idx2 = cr[_i].Idx2;
		_dx = kpl1[_idx1].pt.x - kpl2[_idx2].pt.x;	_dy = kpl1[_idx1].pt.y - kpl2[_idx2].pt.y;
		_val = (_dx*_dx) + (_dy*_dy);
		_dt.at<float>(_i) = sqrt(_val);
	}
	// calculate mean and std. deviation
	cv::Mat _m, _s;
	cv::meanStdDev(_dt, _m, _s);
	float mean = _m.at<float>(0);
	float stddev = _s.at<float>(0);
	float _cutoff = mean + (sigma * stddev);
	// now build output list
	for ( int _i = 0; _i < _n; ++_i ){
		if ( _dt.at<float>(_i) <= _cutoff ){
			crout.push_back(cr[_i]);
		}
	}
}
void filtercorrespondanceransac(const ListIdxCorrespondance& cr,const cv::Mat& _pts3d,const cv::Mat& _pts2d,
								const cv::Matx33f &cam,ListIdxCorrespondance& crout,const int iterations, const float reproj,
								cv::Mat *rvec, cv::Mat *tvec)
{
	cv::Mat _p3, _p2;
	corresponding_rowmat(cr, _pts3d, _pts2d, _p3, _p2);
	int _maxInliers = int(float(cr.size()) * 0.9);
	cv::Mat _inliers;
	cv::Mat _rvec, _tvec;
	crout.clear();
	// use pnp ransac
	if ( tvec == NULL || rvec == NULL ){
		cv::solvePnPRansac(_p3, _p2, cam, cv::Mat(), _rvec, _tvec, false,
			iterations, reproj, _maxInliers, _inliers, CV_P3P);
	} else {
		cv::solvePnPRansac(_p3, _p2, cam, cv::Mat(), *rvec, *tvec, false,
			iterations, reproj, _maxInliers, _inliers, CV_P3P);
	}
	// use inliers to build new crouta
	int _idx;
	for ( int _i = 0; _i < _inliers.rows ; ++_i ){
		_idx = _inliers.at<int>(_i);
		crout.push_back(cr.at(_idx));
	}
	
}
void correspondance_list(const MatchList& matches, ListIdxCorrespondance& cr, const bool flipcorrespondence)
{
	cr.clear();
	IdxCorrespondance _idx;
	if (flipcorrespondence ){
		for ( MatchList::const_iterator iter= matches.begin(); iter < matches.end(); ++iter){
			_idx.Idx2 = iter->queryIdx;
			_idx.Idx1 = iter->trainIdx;
			cr.push_back(_idx);	
		}
	} else { for ( MatchList::const_iterator iter= matches.begin(); iter < matches.end(); ++iter){
		_idx.Idx1 = iter->queryIdx;
		_idx.Idx2 = iter->trainIdx;
		cr.push_back(_idx);	
	}}
}
void corresponding_keypts(const ListIdxCorrespondance& cr, const KPtsList& kp1, const KPtsList& kp2, KPtsList& kp1_m, KPtsList& kp2_m)
{
	kp1_m.clear();
	kp2_m.clear();
	for ( ListIdxCorrespondance::const_iterator iter= cr.begin(); iter < cr.end(); ++iter){
		kp1_m.push_back(kp1[iter->Idx1]);
		kp2_m.push_back(kp2[iter->Idx2]);
	}
}
void corresponding_descriptors(const ListIdxCorrespondance& cr, const cv::Mat& des1, const cv::Mat& des2, cv::Mat& des1_m, cv::Mat& des2_m)
{
	corresponding_rowmat(cr,des1,des2,des1_m, des2_m);
}
void select_descriptorkeypts(const std::vector< int >& _indices, const KPtsList& kpl1, const cv::Mat& des1,
							 KPtsList& kplout, cv::Mat& desout)
{
	int _n = _indices.size();
	desout = cv::Mat(0, des1.cols, des1.type());
	kplout.clear();
	for ( int _i = 0; _i < _n; ++_i){
		kplout.push_back(kpl1[_indices[_i]]);
		desout.push_back(des1.row(_indices[_i]));
	}
}
void corresponding_rowmat(const ListIdxCorrespondance& cr, const cv::Mat& mat1, const cv::Mat& mat2, cv::Mat& out1, cv::Mat& out2)
{
	int _n = cr.size();
	out1 = cv::Mat(0, mat1.cols, mat1.type());
	out2 = cv::Mat(0, mat2.cols, mat2.type());
	for ( int _i = 0; _i < _n; ++_i){
		out1.push_back(mat1.row(cr[_i].Idx1));
		out2.push_back(mat2.row(cr[_i].Idx2));
	}
}
void correspondance_decompose(const ListIdxCorrespondance& list, std::vector< int >& out1, std::vector< int >& out2)
{
	out1.clear();
	out2.clear();
	for ( ListIdxCorrespondance::const_iterator itr = list.begin(); itr < list.end(); ++itr ){
		out1.push_back(itr->Idx1);
		out2.push_back(itr->Idx2);
	}
}
bool inIntList(const int _num, const std::vector< int >& _list)
{
	for ( std::vector<int>::const_iterator itr = _list.begin(); itr < _list.end(); ++itr ){
		if ( *itr == _num ) return true;
	}
	return false;
}
void select_complement_rowmat(const std::vector< int >& rows, const cv::Mat& inmat, cv::Mat& outmat)
{
	outmat = cv::Mat(0, inmat.cols, inmat.type());
	for ( int i = 0; i < inmat.rows; ++i ){
		if ( !(inIntList(i, rows)) ){
			outmat.push_back(inmat.row(i));
		}
	}
}
void select_rowmat(const std::vector< int >& rows, const cv::Mat& inmat, cv::Mat& outmat)
{
	int _n = rows.size();
	outmat = cv::Mat(0, inmat.cols, inmat.type() );
	for ( std::vector<int>::const_iterator itr = rows.begin(); itr < rows.end(); ++itr ){
		outmat.push_back(inmat.row(*itr));
	}
}
void ratiotest_lowe(const ListMatchList& matches, MatchList& good, const float ratio)
{
	good.clear();
	int _n = 0;
	if ( matches.size() > 0 ){
		for ( ListMatchList::const_iterator itr = matches.begin(); itr < matches.end(); ++itr){
			_n = itr->size();
			if ( _n < 2 ){
				good.push_back(itr->at(0));
			} else {
				if ( itr->at(0).distance < ((itr->at(1).distance)*ratio) ){
					good.push_back(itr->at(0));
				}
			}
		}
	}
}
void match_acrossbfl2(const cv::Mat& des, const cv::Mat& des2, ListIdxCorrespondance& list, const float ratio, const int norm)
{
	cv::Ptr<cv::DescriptorMatcher> matcher = new cv::BFMatcher(norm, false);
	match_across(des, des2, list, ratio, matcher);
}
void match_acrossflannl2(const cv::Mat& des, const cv::Mat& des2, ListIdxCorrespondance& list, const float ratio)
{
	// create default flann
	cv::Ptr<cv::flann::IndexParams> indexParams=new cv::flann::KDTreeIndexParams();
	cv::Ptr<cv::flann::SearchParams> searchParams=new cv::flann::SearchParams();
	indexParams->setInt("trees", 6);
	searchParams->setInt("checks", 120);
	cv::Ptr<cv::DescriptorMatcher> matcher = new cv::FlannBasedMatcher(indexParams, searchParams);
	match_across(des, des2, list, ratio, matcher);
}
void match_acrossflannhamming(const cv::Mat& des, const cv::Mat& des2, ListIdxCorrespondance& list, const float ratio)
{
	// create flann object for hamming norm
	cv::Ptr<cv::flann::IndexParams> indexParams= new cv::flann::LshIndexParams(12, 20, 2);
	cv::Ptr<cv::flann::SearchParams> searchParams = new cv::flann::SearchParams();
	//printf("Size of kpts to search = %d x %d\n", des.rows, des2.rows);
	searchParams->setInt("checks", 20);
	cv::Ptr<cv::DescriptorMatcher> matcher = new cv::FlannBasedMatcher(indexParams, searchParams);
	match_across(des, des2, list, ratio, matcher);
}
void match_across(const cv::Mat& des1, const cv::Mat& des2, ListIdxCorrespondance &list,
				  const float ratio, cv::Ptr< cv::DescriptorMatcher > matcher)
{
	MatchList good;
	bool flipcorrespondence = false;
	ListMatchList mtches;
	if ( des1.rows <= des2.rows ){
		matcher->knnMatch(des1, des2, mtches, 2);
	} else {
		matcher->knnMatch(des2, des1, mtches, 2);
		flipcorrespondence = true;
	}
	ratiotest_lowe(mtches, good, ratio);
	correspondance_list(good, list, flipcorrespondence);
}
void point_cloud2d(const KPtsList &kpts, cv::Mat& outpts2d)
{
	int _n = kpts.size();
	outpts2d = cv::Mat(_n, 1, CV_32FC2);
	for ( int _i = 0; _i < _n; ++_i ){
		outpts2d.at<cv::Point2f>(_i) =  kpts.at(_i).pt;
	}
}
void point_cloud2d_64fc1(const KPtsList& kpts, cv::Mat& outpts2d)
{
	int _n = kpts.size();
	outpts2d = cv::Mat(_n, 2, CV_64FC1);
	cv::Point2f _pt;
	for ( int _i = 0; _i < _n; ++_i ){
		_pt = kpts.at(_i).pt;
		outpts2d.at<double>(_i, 0) = float(_pt.x);
		outpts2d.at<double>(_i, 1) = float(_pt.y);
	}	
}
void point_cloud3d(const KPtsList &kpts, const cv::Mat& pts3d, cv::Mat& outpts3d, cv::Mat* outpts2d)
{
	int _n = kpts.size();
	outpts3d = cv::Mat(_n, 1, CV_32FC3);
	cv::Point3f _p3;
	cv::Point2f _p2;
	cv::Point _p;
	if ( outpts2d != NULL ){
		outpts2d->create(_n, 1, CV_32FC2);
		for ( int _i = 0; _i < _n; ++_i ){
			_p2 = kpts.at(_i).pt;
			_p.x = int(_p2.x); _p.y = int(_p2.y);
			_p3 = pts3d.at<cv::Point3f>(_p);
			(*outpts2d).at<cv::Point2f>(_i) = _p2;
			outpts3d.at<cv::Point3f>(_i) = _p3;
		}
	} else {
		for ( int _i = 0, _j=0; _i < _n; ++_i ){
			_p2 = kpts.at(_i).pt;
			_p.x = int(_p2.x); _p.y = int(_p2.y);
			_p3 = pts3d.at<cv::Point3f>(_p);
			outpts3d.at<cv::Point3f>(_i) = _p3;
		}
	}
}
