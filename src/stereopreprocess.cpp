/*
 * Copyright 2016 <Usama Siraj> <osama@smme.edu.pk>
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

#include "stereopreprocess.h"
#include "utils_optim.h"

namespace RiseOdom {

/////////////////////////// Process Image /////////////////////////////////////////////
void StereoPreProcess::ProcessImages(const cv::Mat& imgl, const cv::Mat& imgr)
{
	ListIdxCorrespondance _list;	_list.clear();
	cv::Mat _desc1, _desc2, neighborhood;
	KPtsList _kpts1, _kpts2;
	// compute fast points
	this->fd->detect(imgl, _kpts1);
	this->fd->detect(imgr, _kpts2);
	// compute descriptors
	this->fe->compute(imgl, _kpts1, _desc1);
	this->fe->compute(imgr, _kpts2, _desc2);
	// establish left - right correspondence //
	neighborhood = this->horizontal_scan_neighborhoodMatrix(_kpts1, _kpts2);
	fast_hammingnorm_match_neighborhood(_desc1, _desc2, neighborhood, _list, this->loweratio, this->parscore);
	
	this->data.descriptors = _desc1.clone();
	this->data.points.clear();
	for ( unsigned int _i = 0; _i < _kpts1.size(); ++_i ){
		Stereo_Image_Points impt;
		impt.invalid = true;
		impt.left.x = static_cast<cv::KeyPoint>(_kpts1.at(_i)).pt.x ;
		impt.left.y = static_cast<cv::KeyPoint>(_kpts1.at(_i)).pt.y ;
		this->data.points.push_back(impt);
	}
	for ( int _i = 0; _i < _list.size(); ++_i ){
		int _idx1 = static_cast<IdxCorrespondance>(_list.at(_i)).Idx1 ;		// index of points
		int _idx2 = static_cast<IdxCorrespondance>(_list.at(_i)).Idx2 ;		// index of kpts2
		this->data.points[_idx1].right.x = static_cast<cv::KeyPoint>(_kpts2.at(_idx2)).pt.x;
		this->data.points[_idx1].right.y = static_cast<cv::KeyPoint>(_kpts2.at(_idx2)).pt.y;
		this->data.points[_idx1].updateCamPointHorizontal(this->cam, this->baseline, this->mindisp);
	}
}	
//////////////////////////// Initializers /////////////////////////////////////////////
void StereoPreProcess::initialize(const int bwin, const int maxpts, const float th, const cv::Size imgsize)
{
	int _nx, _ny;
	this->bucketwindow = bwin;
	this->threshold = th;
	if (imgsize.width != 0) _nx = ceil(float(imgsize.width) / float(bwin));
	else _nx = 10;
	if (imgsize.height != 0) _ny = ceil(float(imgsize.height) / float(bwin));
	else _ny = 6;
	this->maxpoints = (int)ceil((float(imgsize.area()) / (720*480)) * maxpts);
	if ( this->maxpoints == 0) this->maxpoints = maxpts;
	this->fd = new cv::GridAdaptedFeatureDetector(new cv::FastFeatureDetector(this->threshold, true), this->maxpoints, _nx, _ny);
	// now initialize descriptors
	this->fe = cv::DescriptorExtractor::create("FREAK");
}
void StereoPreProcess::setCam(const cv::Matx33f& camera, const float bs, const float md)
{
	this->cam = camera;
	this->baseline = bs;
	this->mindisp = md;
}
void StereoPreProcess::setMatching(const int y, const int x, const float lowe, const int ps)
{
	this->_yerr = y;
	this->_xmax = x;
	this->loweratio = lowe;
	this->parscore = ps;
}
//////////////////////////// Get / Set Members /////////////////////////////////////////
float StereoPreProcess::getBaseline() const
{
	return this->baseline;
}
float StereoPreProcess::getMinimumDisparity() const
{
	return this->mindisp;
}
cv::Matx33f StereoPreProcess::getCamIntrinsic() const
{
	return this->cam;
}
Frame_Data StereoPreProcess::getFrameData() const
{
	return this->data;
}
//////////////////////////// neighborhood matrix horizontal ///////////////////////////
cv::Mat StereoPreProcess::horizontal_scan_neighborhoodMatrix(const std::vector< cv::KeyPoint >& kpl,
															 const std::vector< cv::KeyPoint >& kpr) const
{
	float _dist;
	cv::Point2f _diff;
	cv::Mat neigh = cv::Mat::zeros(kpl.size(), kpr.size(), CV_8UC1);
	for ( unsigned int _i = 0; _i < kpl.size(); ++_i){
		for ( unsigned int _j = 0; _j < kpr.size(); ++_j ){
			_diff.x = kpl[_i].pt.x - kpr[_j].pt.x ;
			_diff.y = abs(kpl[_i].pt.y - kpr[_j].pt.y) ;
			if ( _diff.y < this->_yerr ){
				if ( (_diff.x >= 0) && (_diff.x < this->_xmax) ){
					neigh.at<unsigned char>(_i, _j) = 1;
				}
			}
		}
	}
	return neigh;
}
//////////////////////////// Constructors / Destructors ///////////////////////////////
StereoPreProcess::StereoPreProcess()
{
	this->cam << 1., 0, 0, 0, 1., 0, 0.5, 0.5, 0.;
	this->baseline = 1.0;
	this->mindisp = 0.;
	this->initialize(IMGPREPROCESS_DEFAULT_BUCKET, IMGPREPROCESS_DEFAULT_MAXPTS, IMGPREPROCESS_DEFAULT_THRESHOLD, cv::Size(0, 0));
	this->setMatching(IMGPREPROCESS_DEFAULT_YERR, IMGPREPROCESS_DEFAULT_XMAX, IMGPREPROCESS_DEFAULT_LOWE,IMGRPEPROCESS_DEFAULT_PARSCORE);
}
StereoPreProcess::StereoPreProcess(const StereoPreProcess& other)
{
	this->cam = other.cam;
	this->baseline = other.baseline;
	this->mindisp = other.mindisp;
	this->fd = other.fd;
	this->fe = other.fe;
	this->_xmax = other._xmax;
	this->_yerr = other._yerr;
	this->loweratio = other.loweratio;
	this->parscore = other.parscore;
	this->bucketwindow = other.bucketwindow;
	this->threshold = other.threshold;
	this->maxpoints = other.maxpoints;
	
	this->data = other.data;
}
StereoPreProcess::~StereoPreProcess()
{ }
StereoPreProcess& StereoPreProcess::operator=(const StereoPreProcess& other)
{
	this->cam = other.cam;
	this->baseline = other.baseline;
	this->mindisp = other.mindisp;
	this->fd = other.fd;
	this->fe = other.fe;
	this->_xmax = other._xmax;
	this->_yerr = other._yerr;
	this->loweratio = other.loweratio;
	this->parscore = other.parscore;
	this->bucketwindow = other.bucketwindow;
	this->threshold = other.threshold;
	this->maxpoints = other.maxpoints;
	
	this->data = other.data;
	
	return *this;
}
//////////////////////////////// Public Helper Functions ///////////////////////////////
cv::Point2f leftcam2imageplane(const cv::Matx33f& cam, const cv::Point3f& cam_point)
{
	cv::Point3f _puv = cam * cam_point;
	cv::Point2f _uv;
	_uv.x = _puv.x / _puv.z ;	_uv.y = _puv.y / _puv.z;
	return _uv;
}
cv::Point2f frameN2imageplane(const cv::Matx33f &cam, const TransformationMatrix &Cam2FrameN, const cv::Point3f &frameN_point){
	return leftcam2imageplane(cam, Cam2FrameN.transformPoint(frameN_point));
}

Stereo_Image_Points cam2stereoimageplane(const cv::Matx33f& cam, const float _baseline,const cv::Point3f& cam_point, bool horizontal)
{
	Stereo_Image_Points _pt;
	if ( cam_point.z < 0. ){
		_pt.invalid = true;
		return _pt;
	}	
	cv::Point3f _puv = cam * cam_point;
	float _td = _puv.z / cam(0,0);
	float _disp = _baseline / _td;
	_pt.left.x = _puv.x / _puv.z ;	_pt.left.y = _puv.y / _puv.z;
	_pt.right.x = _pt.left.x;	_pt.right.y = _pt.left.y;
	if ( horizontal ) _pt.right.x -= _disp;
	else _pt.right.y -= _disp;
	_pt.leftcamPoint = cam_point;
	_pt.invalid = false;
	return _pt;
}
Stereo_Image_Points frameN2stereoimageplane(const cv::Matx33f& cam, const float _baseline,
		const TransformationMatrix& Cam2FrameN, const cv::Point3f& frameN_point, bool horizontal)
{
	return cam2stereoimageplane(cam, _baseline, Cam2FrameN.transformPoint(frameN_point), horizontal);
}
/////////////////////////////// Stereo Image Points /////////////////////////////////
Stereo_Image_Points& Stereo_Image_Points::operator=(const Stereo_Image_Points& other)
{
	this->left = other.left;
	this->right = other.right;
	this->leftcamPoint = other.leftcamPoint;
	this->invalid = other.invalid;
	return *this;
}
Stereo_Image_Points::Stereo_Image_Points(const Stereo_Image_Points& other)
{
	this->left = other.left;
	this->right = other.right;
	this->leftcamPoint = other.leftcamPoint;
	this->invalid = other.invalid;
}
Stereo_Image_Points::Stereo_Image_Points()
{
	this->left.x = 0.;	this->right.x = 0;
	this->left.y = 0.;	this->right.y = 0;
	this->leftcamPoint.x = 0.;	this->leftcamPoint.y = 0.;	this->leftcamPoint.z = 0.;
	this->invalid = true;
}
Stereo_Image_Points::~Stereo_Image_Points()
{ }
void Stereo_Image_Points::updateCamPointHorizontal(const cv::Matx33f& cam, const float _baseline, const float _mindisp)
{
	float _disp = this->left.x - this->right.x ;
	if ( _disp > _mindisp ){
		float _td = _baseline / _disp;
		this->leftcamPoint.z = (cam(0,0) * _td);
		this->leftcamPoint.y = (this->left.y - cam(1,2)) * _td;
		this->leftcamPoint.x = (this->left.x - cam(0,2)) * _td;
		this->invalid = false;
	} else {
		this->invalid = true;
	}
}
void Stereo_Image_Points::updateCamPointVertical(const cv::Matx33f& cam, const float _baseline, const float _mindisp)
{
	float _disp = this->left.y - this->right.y ;
	if ( _disp > _mindisp ){
		float _td = _baseline / _disp;
		this->leftcamPoint.z = (cam(0,0) * _td);
		this->leftcamPoint.y = (this->left.y - cam(1,2)) * _td;
		this->leftcamPoint.x = (this->left.x - cam(0,2)) * _td;
		this->invalid = false;
	} else {
		this->invalid = true;
	}
}
void Stereo_Image_Points::print_screen() const
{
	if ( this->invalid ) printf("Invalid Stereo Image Point\n");
	else printf("Left :%lf,%lf -- Right : %lf,%lf -- Left Cam Point : %lf,%lf,%lf\n", this->left.x, this->left.y,
		   this->right.x, this->right.y, this->leftcamPoint.x, this->leftcamPoint.y, this->leftcamPoint.z );
}
////////////////////////////// Frame Data //////////////////////////////////////////
Frame_Data::Frame_Data(const Frame_Data& other)
{
	this->frame_id = other.frame_id;
	this->descriptors = cv::Mat();
	other.descriptors.copyTo(this->descriptors);
	this->points = other.points;
}
Frame_Data& Frame_Data::operator=(const Frame_Data& other)
{
	this->frame_id = other.frame_id;
	this->descriptors = cv::Mat();
	other.descriptors.copyTo(this->descriptors);
	this->points = other.points;
	
	return *this;
}
Frame_Data::Frame_Data()
{
	this->frame_id = -1;
	this->descriptors = cv::Mat();
	this->points.clear();
}
Frame_Data::~Frame_Data()
{
	this->frame_id = -1;
	this->descriptors = cv::Mat();
	this->points.clear();
}
void Frame_Data::print_screen() const
{
	printf("Frame ID : %d\n" , this->frame_id );
	if ( this->frame_id != -1 ){
		for ( unsigned int _i = 0; _i < this->points.size(); ++_i ){
			this->points[_i].print_screen();
		}
	}
}
Frame_Data Frame_Data::filter_invalid() const
{
	Frame_Data res;
	res.frame_id = this->frame_id;
	res.descriptors = cv::Mat(0, this->descriptors.cols, this->descriptors.type());
	for ( unsigned int _i = 0; _i < this->points.size(); ++_i ){
		if ( !(this->points[_i].invalid) ){
			res.descriptors.push_back(this->descriptors.row(_i));
			res.points.push_back(this->points[_i]);
		}
	}
	return res;
}
Frame_Data Frame_Data::getMatched(const std::vector< int >& mtchs, bool islhs) const
{
	if ( islhs ) {
		CV_Assert(mtchs.size() == this->descriptors.rows );
	}
	Frame_Data rdata;
	rdata.frame_id = this->frame_id;
	rdata.descriptors = cv::Mat(0, this->descriptors.cols, this->descriptors.type());
	for ( int i = 0; i < mtchs.size(); ++i ){
		if ( mtchs[i] != -1 ){
			if ( islhs ) {
				rdata.descriptors.push_back(this->descriptors.row(i));
				rdata.points.push_back(this->points[i]);
			} else {
				rdata.descriptors.push_back(this->descriptors.row(mtchs[i]));
				rdata.points.push_back(this->points[mtchs[i]]);
			}
		}
	}
	return rdata;
}
Frame_Data Frame_Data::predicted_data(const TransformationMatrix& trmat, const cv::Matx33f &cam, const float _baseline) const
{
	// Compute Predicted Frame Data based on current data , trmat is the transformation of camera frame
	Frame_Data out = *this;	// copy all points
	out.predict(trmat, cam, _baseline);
	return out;
}
void Frame_Data::predict(const TransformationMatrix& trmat, const cv::Matx33f& cam, const float _baseline)
{
	for ( int i = 0; i < this->points.size(); ++i ){
		if ( !this->points[i].invalid ){
			cv::Point3f projectedpoint =  trmat.transformPoint(this->points[i].leftcamPoint);
			this->points[i] = cam2stereoimageplane(cam, _baseline, projectedpoint);
		}
	}
}
cv::Mat Frame_Data::getPts2D() const
{
	cv::Mat _pts2D(this->points.size(), 1, CV_32FC2);
	for ( unsigned int _i = 0; _i < this->points.size(); ++_i ){
		_pts2D.at<cv::Point2f>(_i) = static_cast<Stereo_Image_Points>(this->points[_i]).left;
	}
	return _pts2D;
}
cv::Mat Frame_Data::getPts3D() const
{
	cv::Mat _pts3D(this->points.size(), 1, CV_32FC3);
	for ( unsigned int _i = 0; _i < this->points.size(); ++_i ){
		_pts3D.at<cv::Point3f>(_i) = static_cast<Stereo_Image_Points>(this->points[_i]).leftcamPoint;
	}
	return _pts3D;
}
int Frame_Data::getValidCount() const
{
	int cnt = 0;
	for ( int _i = 0; _i < this->points.size(); ++_i ){
		if ( !(this->points[_i].invalid) ) cnt += 1;
	}
	return cnt;
}
std::vector< int > Frame_Data::match_with(const Frame_Data& other, const float loweratio,
										  const int parscore,const float maxradius) const
{
	ListIdxCorrespondance _list;
	std::vector<int> mres;	mres.resize(this->descriptors.rows);
	for ( unsigned int _i = 0; _i < mres.size(); ++_i ){
		mres[_i] = -1;		// intialize to -1 i.e no match found
	}
	if ( maxradius > 0. ){	
		cv::Mat neighborhood = this->neighborhood_matrix(other, maxradius);
		fast_hammingnorm_match_neighborhood(this->descriptors, other.descriptors, neighborhood , _list, loweratio, parscore);
	} else {
		fast_hammingnorm_match(this->descriptors, other.descriptors, _list, loweratio, parscore);
	}
	for ( unsigned int _i = 0; _i < _list.size(); ++_i ){
		mres[ static_cast<IdxCorrespondance>(_list[_i]).Idx1 ] = static_cast<IdxCorrespondance>(_list[_i]).Idx2;
	}
	return mres;
}
std::vector< int > Frame_Data::match_with_ransac(const Frame_Data& other, const float loweratio, const int parscore,
												 const float maxradius, const cv::Matx33f &cam, const float reprojerrors,
												 const int max_iterations, cv::Mat *rvec, cv::Mat *tvec) const
{
	std::vector<int> initial_match = this->match_with(other, loweratio, parscore, maxradius);
	std::vector<int> initial_match_idcs;
	cv::Mat _pts2d(0,1,CV_32FC2), _pts3d(0,1,CV_32FC3);
	for ( unsigned int _i = 0; _i < initial_match.size(); ++_i ){
		if ( initial_match[_i] != -1 ){
			_pts3d.push_back(this->points[_i].leftcamPoint);
			_pts2d.push_back(other.points[initial_match[_i]].left);
			initial_match_idcs.push_back(_i);
		}
	}
	if ( _pts2d.rows > 5 ){
		int _maxInliers = int(float(_pts2d.rows) * 0.9);
		cv::Mat _inliers;
		if ( tvec == NULL || rvec == NULL ){
			cv::Mat _rvec, _tvec;
			cv::solvePnPRansac(_pts3d, _pts2d, cam, cv::Mat(), _rvec, _tvec, false,
					max_iterations, reprojerrors, _maxInliers, _inliers, CV_P3P);
		} else {
			cv::solvePnPRansac(_pts3d, _pts2d, cam, cv::Mat(), *rvec, *tvec, false,
					max_iterations, reprojerrors, _maxInliers, _inliers, CV_P3P);
		}
		////////////////// now remove all except inliers /////////////////////////
		std::vector<int> final_match;	final_match.resize(this->descriptors.rows);
		for ( unsigned int _i = 0; _i < final_match.size(); ++_i ){
			final_match[_i] = -1;		// intialize to -1 i.e no match found
		}
		for ( int _i = 0; _i < _inliers.rows; ++_i ){
			int _idx =  initial_match_idcs[_inliers.at<int>(_i)];
			final_match[_idx] = initial_match[_idx];
		}
		return final_match;
	}
	return initial_match;	
}
TransformationMatrix Frame_Data::estimate_delta_motion(const Frame_Data& other, const std::vector< int >& match_list,
													   const cv::Matx33f &cam,  cv::Mat *rvec, cv::Mat *tvec) const
{
	TransformationMatrix transform;
	cv::Mat _pts2d(0,1,CV_32FC2), _pts3d(0,1,CV_32FC3);
	for ( unsigned int _i = 0; _i < match_list.size(); ++_i ){
		if ( match_list[_i] != -1 ){
			_pts3d.push_back(this->points[_i].leftcamPoint);
			_pts2d.push_back(other.points[match_list[_i]].left);
		}
	}
	if ( _pts2d.rows > 5 ){
		if (  tvec == NULL || rvec == NULL  ) {
			cv::Mat _rvec, _tvec;
			cv::solvePnP(_pts3d, _pts2d, cam, cv::Mat(), _rvec, _tvec, false, CV_ITERATIVE);
			transform.setRodrigues(_rvec, _tvec);
			transform.inverse();
		} else {			
			cv::solvePnP(_pts3d, _pts2d, cam, cv::Mat(), *rvec, *tvec, true, CV_ITERATIVE);
			transform.setRodrigues(*rvec, *tvec);
			transform.inverse();
		}
	}
	return transform;
}
cv::Mat Frame_Data::neighborhood_matrix(const Frame_Data& other, const float _maxradius) const
{
	float _dist;
	cv::Point2f _diff;
	cv::Mat _neighmat = cv::Mat::zeros(this->descriptors.rows, other.descriptors.rows, CV_8UC1);
	for ( unsigned int _i = 0; _i < this->points.size(); ++_i ){
		for ( unsigned int _j = 0; _j < other.points.size(); ++_j ){
			_diff.x = this->points[_i].left.x - other.points[_j].left.x;	_diff.x *= _diff.x;
			_diff.y = this->points[_i].left.y - other.points[_j].left.y;	_diff.y *= _diff.y;
			_dist = sqrt(_diff.x + _diff.y );
			if ( _dist <= _maxradius ){
				_neighmat.at<unsigned char>(_i, _j) = 1;
			}
		}
	}
	return _neighmat;
}
cv::Mat draw_framescanmatch(const cv::Mat& imgl, const cv::Mat& imgr, const Frame_Data& fd)
{
	cv::Mat outimage;
	int _cnt = 0;
	cv::addWeighted(imgl, 0.5, imgr, 0.5, 0., outimage);
	if (outimage.channels() == 1){
		cv::Mat arr[3];
		arr[0] = outimage; arr[1] = outimage; arr[2] = outimage;
		cv::merge(arr, 3, outimage);
	}
	for ( std::vector<Stereo_Image_Points>::const_iterator itr = fd.points.begin(); itr < fd.points.end(); ++itr ){
		if ( !( itr->invalid) ){
			cv::line(outimage, itr->left, itr->right, cv::Scalar(10,10,255),1);
			cv::circle(outimage, itr->right, 2, cv::Scalar(255, 10, 10), 1);
			cv::circle(outimage, itr->left, 2, cv::Scalar(10, 2555, 10), 1);
			_cnt += 1;
			char _zdist[16];
			sprintf(_zdist, "%.0f", itr->left.x - itr->right.x);
			cv::Point _txtpt;
			_txtpt.x = (itr->left.x + itr->right.x) / 2.;
			_txtpt.y = (itr->left.y + itr->right.y) / 2.;
			cv::putText(outimage, _zdist, _txtpt, cv::FONT_HERSHEY_SCRIPT_COMPLEX, 0.25, cv::Scalar(10,255,255), 1);			
		}
	}
	//////////////////// Frame No. ///////////////////////////////////
	char _txt[32];
	sprintf(_txt,"Frame No. %d, 3D Points : %d", fd.frame_id, _cnt);
	cv::putText(outimage, _txt, cv::Point((outimage.cols / 2) - 80, 40), cv::FONT_HERSHEY_PLAIN,
		1.0, cv::Scalar(20, 230, 120), 2);	
	return outimage;
}
cv::Mat draw_framedata(const cv::Mat& img, const Frame_Data& fd, int clr_scheme, bool numbering)
{
	cv::Mat outimage;
	if (img.channels() == 1){
		cv::Mat arr[3];
		arr[0] = img; arr[1] = img; arr[2] = img;
		cv::merge(arr, 3, outimage);
	} else {
		outimage = img.clone();
	}
	cv::Scalar clr1, clr2;
	switch ( clr_scheme ){
		case 1:
			clr1 = cv::Scalar(255, 10, 10);
			clr2 = cv::Scalar(128, 10, 10);
			break;
		case 2:
			clr1 = cv::Scalar(10, 10 , 255);
			clr2 = cv::Scalar(10, 10, 128);
			break;
		case 3:
			clr1 = cv::Scalar(10, 255, 10);
			clr2 = cv::Scalar(10, 128, 10);
			break;
		default:
			clr1 = cv::Scalar(200, 200, 200);
			clr2 = cv::Scalar(50, 50, 50);
			break;
	}
	for ( std::vector<Stereo_Image_Points>::const_iterator itr = fd.points.begin(); itr < fd.points.end(); ++itr ){
		if ( itr->invalid ){
			cv::circle(outimage, itr->left, 3, clr2, 1);
		} else {			
			////// print distance
			if ( numbering ){
				char _zdist[16];
				sprintf(_zdist, "%.2lf", itr->leftcamPoint.z);
				cv::putText(outimage, _zdist, itr->left, cv::FONT_HERSHEY_PLAIN, 0.45, clr2, 1);		
			} else {
				cv::circle(outimage, itr->left, 1, clr1, 2);
			}
		}
	}
	return outimage;
}
cv::Mat draw_matched_framedata(const cv::Mat& img, const Frame_Data& fd1, const Frame_Data& fd2,
							   const std::vector< int >& mtchs)
{
	cv::Mat outimage;
	if (img.channels() == 1){
		cv::Mat arr[3];
		arr[0] = img; arr[1] = img; arr[2] = img;
		cv::merge(arr, 3, outimage);
	} else {
		outimage = img.clone();
	}
	////////////////////// Draw Key Points ////////////////////////////
	outimage = draw_framedata(outimage, fd1, 1);
	outimage = draw_framedata(outimage, fd2, 2, true);
	int _cnt = 0;
	for ( unsigned int _i = 0; _i < mtchs.size(); ++_i ){
		if ( mtchs[_i] != -1 ){
			cv::line(outimage, fd1.points[_i].left, fd2.points[mtchs[_i]].left, cv::Scalar(10, 250, 250), 2);
			_cnt += 1;
		}
	}
	//////////////////// Frame No. ///////////////////////////////////
	char _txt[32];
	sprintf(_txt,"Frame No. %d, Matches : %d", fd2.frame_id, _cnt);
	cv::putText(outimage, _txt, cv::Point((outimage.cols / 2) - 80, 20), cv::FONT_HERSHEY_PLAIN,
		1.0, cv::Scalar(20, 230, 120), 2);
	return outimage;
}
cv::Mat draw_transforminfo(const cv::Mat& img, const TransformationMatrix& trmat, const double frwdvel, const double angvel)
{
	cv::Matx44d _trmat = trmat.getMatrix();
	cv::Mat _outimage = img.clone();
	char _line1[64], _line2[64], _line3[64], _line4[64];
	sprintf(_line1, "%8.3lf -- %6.3lf", frwdvel, angvel);
	sprintf(_line2, "%5.2lf,%5.2lf,%5.2lf |%5.2lf", _trmat(0,0), _trmat(0,1), _trmat(0,2), _trmat(0,3));
	sprintf(_line3, "%5.2lf,%5.2lf,%5.2lf |%5.2lf", _trmat(1,0), _trmat(1,1), _trmat(1,2), _trmat(1,3));
	sprintf(_line4, "%5.2lf,%5.2lf,%5.2lf |%5.2lf", _trmat(2,0), _trmat(2,1), _trmat(2,2), _trmat(2,3));
	
	cv::Point _stcoord;
	_stcoord.x = 15;
	_stcoord.y = img.rows - 120;
	cv::putText(_outimage, _line1, _stcoord, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(10, 10, 10), 2 );	
	cv::putText(_outimage, _line1, _stcoord, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 200, 200), 1 );	_stcoord.y += 25;
	cv::putText(_outimage, _line2, _stcoord, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(10, 10, 10), 2 );
	cv::putText(_outimage, _line2, _stcoord, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 200, 200), 1 );	_stcoord.y += 25;
	cv::putText(_outimage, _line3, _stcoord, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(10, 10, 10), 2 );
	cv::putText(_outimage, _line3, _stcoord, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 200, 200), 1 );	_stcoord.y += 25;
	cv::putText(_outimage, _line4, _stcoord, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(10, 10, 10), 2 );
	cv::putText(_outimage, _line4, _stcoord, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 200, 200), 1 );
	
	return _outimage;
}


}