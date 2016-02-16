/*
 * image_database.cpp
 *
 *  Created on: Feb 5, 2015
 *      Author: usama
 */

#include "image_database.h"
#include <stdio.h>
#include <fstream>
#include <dirent.h>

KittiDatabase::KittiDatabase(const unsigned int seq, const char* basedir){
	this->dataset = seq;
	this->_build_base_db(basedir);
	this->_load_timings();
	this->_load_calibration();
	if ( this->_pose_exist()){
		this->ground_truth = load_poses(this->_getfile_pose().c_str());
	}
}
KittiDatabase::~KittiDatabase() {
}
// utility functions
void KittiDatabase::_build_base_db(const char *basedir){
	this->dir_base = basedir;
	this->dir_pose = "poses/";
	this->dir_image = "sequences/%02d/";
	this->type_image = "%06d.png";
	this->type_pose = "%02d.txt";
	this->left = "image_0/";
	this->right = "image_1/";
	this->calibfile = "calib.txt";
	this->timefile = "times.txt";
	for (unsigned int i = 0; i < 11; ++i){
		this->poses.push_back(i);
	}
	for (unsigned int i = 0; i < 22; ++i){
		this->sequences.push_back(i);
	}
	this->lastimage.push_back(4540);
	this->lastimage.push_back(1100);
	this->lastimage.push_back(4660);
	this->lastimage.push_back(800);
	this->lastimage.push_back(270);
	this->lastimage.push_back(2760);
	this->lastimage.push_back(1100);
	this->lastimage.push_back(1100);
	this->lastimage.push_back(4070);
	this->lastimage.push_back(1590);
	this->lastimage.push_back(1200);
	this->lastimage.push_back(920);
	this->lastimage.push_back(1060);
	this->lastimage.push_back(3280);
	this->lastimage.push_back(630);
	this->lastimage.push_back(1900);
	this->lastimage.push_back(1730);
	this->lastimage.push_back(490);
	this->lastimage.push_back(1800);
	this->lastimage.push_back(4980);
	this->lastimage.push_back(830);
	this->lastimage.push_back(2720);
}
bool KittiDatabase::_dbnum_exist() const{
	for (std::vector<unsigned int>::const_iterator it = this->sequences.begin(); it < this->sequences.end(); ++it){
		if ( *it == this->dataset){
			return true;
		}
	}
	return false;
}
bool KittiDatabase::_pose_exist() const{
	for (std::vector<unsigned int>::const_iterator it = this->poses.begin(); it < this->poses.end(); ++it){
		if (*it == this->dataset){
			return true;
		}
	}
	return false;
}
bool KittiDatabase::_imgseq_exist(const unsigned int _seq) const{
	if (_seq <= this->lastimage[this->dataset]){
		return true;
	}
	return false;
}
void KittiDatabase::_load_calibration(){
	std::string _flname = this->_getfile_callib();
	std::ifstream _file;
	_file.open(_flname.c_str());
	if ( !_file ){
		fprintf(stderr, "Calibration file not found!");
		return;
	}
	std::vector<double> nlist;
	char _snum[64];
	double _num;
	_file >> _snum;
	while ( !_file.eof()){
		_num = -5000;
		sscanf(_snum,"%lf", &_num);
		if (_num != -5000){
			nlist.push_back(_num);
		}
		_file >> _snum;
	}
	_file.close();
	if ( nlist.size() < 24){
		fprintf(stderr, "Calib file not properly read!");
		return;
	}
	// now build appropriate matrices
	cv::Matx34d P0, P1, P0n, P1n;
	for ( unsigned int _i = 0; _i < 12; ++_i ){
		unsigned int _c = _i % 4;
		unsigned int _r = _i / 4;
		P0(_r, _c) = nlist[_i];
		P1(_r, _c) = nlist[_i+12];		
	}
	double _lfx, _lfy, _lcx, _lcy, _rfx, _rfy, _rcx, _rcy;
	_lfx = P0(0, 0);	_lfy = P0(1, 1); _lcx = P0(0, 2); _lcy = P0(1, 2);
	_rfx = P1(0, 0);	_rfy = P1(1, 1); _rcx = P1(0, 2); _rcy = P1(1, 2);
	cv::Matx33d K0, K1, K0inv, K1inv;
	K0 << _lfx, 0, _lcx, 0, _lfy, _lcy, 0, 0, 1;
	K1 << _rfx, 0, _rcx, 0, _rfy, _rcy, 0, 0, 1;
	K0inv << 1./_lfx, 0, -_lcx/_lfx, 0, 1./_lfy, -_lcy/_lfy, 0, 0, 1;
	K1inv << 1./_rfx, 0, -_rcx/_rfx, 0, 1./_rfy, -_rcy/_rfy, 0, 0, 1;
	P0n = K0inv * P0;
	P1n = K1inv * P1;
	this->Kleft = K0;
	this->Pleft = P0n;
	this->Kright = K1;
	this->Pright = P1n;
}
// other helping

std::vector<cv::Matx34d> load_poses(const char *filename){
	std::vector<cv::Matx34d> _resp;
	std::ifstream _file(filename);
	std::string _line;
	if ( !_file ){
		fprintf(stderr, "Pose File : %s doesn't exist", filename);
	} else {
		while (std::getline(_file, _line)){
			std::vector<double> _nums;
			double _num;
			std::istringstream  iss(_line);			
			while (!iss.eof()){
				iss >> _num;
				_nums.push_back(_num);
			}
			if (_nums.size() >= 12){
				cv::Matx34d _tmat;
				for ( unsigned int _i = 0; _i < 12; ++_i ){
					unsigned int _c = _i % 4;
					unsigned int _r = _i / 4;
					_tmat(_r, _c) = _nums[_i];	
				}
				_resp.push_back(_tmat);
			}			
		}		
		_file.close();
	}	
	return _resp;
}

void save_poses(const char *filename, const std::vector<cv::Matx34d> &_poses){
	std::ofstream _file(filename);
	if ( !_file ) fprintf(stderr, "Cannot write pose to file : %s\n", filename);
	else {
		_file << std::scientific ;
		for ( std::vector<cv::Matx34d>::const_iterator itr = _poses.begin(); itr < _poses.end(); ++itr ){
			for ( unsigned int _i = 0; _i < 12; ++_i ){
				unsigned int _c = _i % 4;
				unsigned int _r = _i / 4;
				_file << (*itr)(_r, _c) << " ";
			}
			_file << std::endl;
		}		
	}
	_file.close();
}
/*
 * CLASS : Image Database
 * 
 * To Access image sequences stored on disk for offline processing
 */
// Constructors / Destructors
ImageDatabase::ImageDatabase()
{	
	this->scalingfactor = 1.;
}
ImageDatabase::ImageDatabase(const unsigned int seq, const char* basedir)
:dataset(seq){
	this->scalingfactor = 1.;
	this->_build_base_db(basedir);
	this->_load_timings();
	this->_load_calibration();
	if ( this->_pose_exist()){
		this->ground_truth = load_poses(this->_getfile_pose().c_str());
	}
}
ImageDatabase::~ImageDatabase(){
}
bool ImageDatabase::isImageFileEmpty(const std::vector< std::string > &_name)
{
	if ( (_name.size() < 2) || (_name[0] == "") || (_name[1] == "") ){
		return true;
	}
	return false;
}
// member functions
cv::Matx34d ImageDatabase::getGroundTruth(const int _seq) const
{
	cv::Matx34d _res;
	_res << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
	if ( this->_pose_exist() && this->_imgseq_exist(_seq) ){
		_res = this->ground_truth[_seq];		
	}
	return _res;
}
double ImageDatabase::get_baseline() const
{
	if ( this->Pright(0, 3) < 0 ) return -(this->Pright(0, 3));
	return this->Pright(0, 3);
}
double ImageDatabase::get_focus() const
{
	return this->Kleft(0, 0);
}
cv::Matx34d ImageDatabase::getExtrinsicLeft() const
{
	return this->Pleft;
}
cv::Matx34d ImageDatabase::getExtrinsicRight() const
{
	return this->Pright;
}
cv::Matx33d ImageDatabase::getIntrinsicLeft() const
{
	return this->Kleft;
}
cv::Matx33d ImageDatabase::getIntrinsicRight() const
{
	return this->Kright;
}
bool ImageDatabase::load_image(const unsigned int _seq, cv::Mat* imleft, cv::Mat* imright, double* timing) const
{
	if (this->_imgseq_exist(_seq)){
		std::vector<std::string> _flname = this->get_image(_seq, timing);
		if ( imleft != NULL){
			(*imleft) = cv::imread(_flname[0]);
			if ( this->scalingfactor != 1. ){
				cv::resize(*imleft, *imleft, cv::Size(), this->scalingfactor, this->scalingfactor, cv::INTER_LINEAR);
			}
		}
		if ( imright != NULL){
			(*imright) = cv::imread(_flname[1]);
			if ( this->scalingfactor != 1. ){
				cv::resize(*imright, *imright, cv::Size(), this->scalingfactor, this->scalingfactor, cv::INTER_LINEAR);
			}
		}
		return true;
	} else {
		return false;
	}
}
bool ImageDatabase::load_grayimage(const unsigned int _seq, cv::Mat* imleft, cv::Mat* imright, double* timing) const
{
	bool _rval = this->load_image(_seq, imleft, imright, timing);
	if ( _rval ){
		if ((imleft != NULL) && (imleft->channels() != 1) ){
			cv::cvtColor(*imleft, *imleft, cv::COLOR_BGR2GRAY);
		}
		if ( (imright != NULL) && (imright->channels() != 1)){
			cv::cvtColor(*imright, *imright, cv::COLOR_BGR2GRAY);
		}		
	}	
	return _rval;
	
}
std::vector< std::string > ImageDatabase::get_image(const unsigned int _seq, double* timing) const
{
	std::vector<std::string> _res;
	if ( timing != NULL){
		if (this->_imgseq_exist(_seq)){
			*timing = double(this->timings[_seq]);
		} else {
			*timing = -1;
		}
	}
	if ( this->_dbnum_exist() && this->_imgseq_exist(_seq)){
		_res = this->_getfile_image(_seq);
	} else {
		_res.push_back("");
		_res.push_back("");
	}
	return _res;
	
}
cv::Size ImageDatabase::image_size() const
{
	cv::Mat _img;
	this->load_grayimage(0, &_img);
	cv::Size _rs(_img.cols, _img.rows);
	return _rs;
}
bool ImageDatabase::is_groundtruth_available() const
{
	return this->_pose_exist();
}
double ImageDatabase::get_scale() const
{
	return this->scalingfactor;
}
void ImageDatabase::set_scale(const double scale)
{
	if ( scale == 0 ) return;
	// normalize
	this->_readjustintrinsic(1./this->scalingfactor);
	// set scale
	this->_readjustintrinsic(scale);
	this->scalingfactor = scale;
}
// utility functions
std::string ImageDatabase::_getfile_callib() const
{
	char _imgdir[64];
	snprintf(_imgdir, 64, this->dir_image.c_str(), this->dataset);
	std::string _calibfile = this->dir_base + std::string(_imgdir) + this->calibfile;
	return _calibfile;
}
std::string ImageDatabase::_getfile_pose() const
{
	char _posfil[64];
	snprintf(_posfil, 64, this->type_pose.c_str(), this->dataset);
	std::string _posefile = this->dir_base + this->dir_pose + std::string(_posfil);
	return _posefile;
}
std::string ImageDatabase::_getfile_time() const
{
	char _imgdir[64];
	snprintf(_imgdir, 64, this->dir_image.c_str(), this->dataset);
	std::string _tmfile = this->dir_base + std::string(_imgdir) + this->timefile;
	return _tmfile;
}
std::vector< std::string > ImageDatabase::_getfile_image(const unsigned int _seq) const
{
	std::vector<std::string> _res;
	char _imgdir[64];
	char _imgfil[64];
	snprintf(_imgdir, 64, this->dir_image.c_str(), this->dataset);
	snprintf(_imgfil, 64, this->type_image.c_str(), _seq);
	std::string _simgfil(_imgfil);
	std::string _simgdir(_imgdir);
	std::string _leftimg = this->dir_base + _simgdir + this->left + _simgfil;
	std::string _rightimg = this->dir_base + _simgdir + this->right + _simgfil;
	_res.push_back(_leftimg);
	_res.push_back(_rightimg);
	return _res;
}
bool ImageDatabase::_dbnum_exist() const
{
	char _imgdir[64];
	snprintf(_imgdir, 64, this->dir_image.c_str(), this->dataset);
	std::string _simgdir(_imgdir), _sdbdir;
	_sdbdir = this->dir_base + _simgdir;
	DIR* dir = opendir(_sdbdir.c_str());
	if (dir){
		/* Directory exists. */
		closedir(dir);
		return true;
	} else {
		return false;
	}
}
bool ImageDatabase::_imgseq_exist(const unsigned int _seq) const
{
	std::vector<std::string> _flname = this->_getfile_image(_seq);
	std::ifstream _ifile;
	_ifile.open(_flname[0].c_str());
	if ( !_ifile ) return false;
	_ifile.close();
	_ifile.open(_flname[1].c_str());
	if ( !_ifile ) return false;
	_ifile.close();
	return true;
}
bool ImageDatabase::_pose_exist() const
{
	std::string _flname = this->_getfile_pose();
	std::ifstream _ifile;
	_ifile.open(_flname.c_str());
	if ( !_ifile ) return false;
	_ifile.close();	
	return true;
}
void ImageDatabase::_load_timings()
{
	std::string _flname = this->_getfile_time();
	std::ifstream _file;
	_file.open(_flname.c_str());
	if ( !_file ){
		fprintf(stderr, "Timing file not found!");
		return;
	}
	float _num;
	_file >> _num;
	while (!_file.eof()){
		this->timings.push_back(_num);
		_file >> _num;
	}
	_file.close();
}
void ImageDatabase::_load_calibration()
{
	std::string _flname = this->_getfile_callib();
	std::ifstream _file;
	_file.open(_flname.c_str());
	if ( !_file ){
		fprintf(stderr, "Calibration file not found!");
		return;
	}
	std::vector<double> nlist;
	char _snum[64];
	double _num;
	_file >> _snum;
	while ( !_file.eof()){
		_num = -5000;
		sscanf(_snum,"%lf", &_num);
		if (_num != -5000){
			nlist.push_back(_num);
		}
		_file >> _snum;
	}
	_file.close();
	if ( nlist.size() < 42){
		fprintf(stderr, "Calib file not properly read!");
		return;
	}
	// now build appropriate matrices
	cv::Matx34d P0, P1, P0n, P1n;
	const int offset = 18;
	for ( unsigned int _i = 0; _i < 12; ++_i ){
		unsigned int _c = _i % 4;
		unsigned int _r = _i / 4;
		P0(_r, _c) = nlist[offset + _i];
		P1(_r, _c) = nlist[offset + _i+12];
	}
	double _lfx, _lfy, _lcx, _lcy, _rfx, _rfy, _rcx, _rcy;
	_lfx = P0(0, 0);	_lfy = P0(1, 1); _lcx = P0(0, 2); _lcy = P0(1, 2);
	_rfx = P1(0, 0);	_rfy = P1(1, 1); _rcx = P1(0, 2); _rcy = P1(1, 2);
	cv::Matx33d K0, K1, K0inv, K1inv;
	K0 << _lfx, 0, _lcx, 0, _lfy, _lcy, 0, 0, 1;
	K1 << _rfx, 0, _rcx, 0, _rfy, _rcy, 0, 0, 1;
	K0inv << 1./_lfx, 0, -_lcx/_lfx, 0, 1./_lfy, -_lcy/_lfy, 0, 0, 1;
	K1inv << 1./_rfx, 0, -_rcx/_rfx, 0, 1./_rfy, -_rcy/_rfy, 0, 0, 1;
	P0n = K0inv * P0;
	P1n = K1inv * P1;
	this->Kleft = K0;
	this->Pleft = P0n;
	this->Kright = K1;
	this->Pright = P1n;
}
void ImageDatabase::_build_base_db(const char* _basedir)
{
	this->dir_base = _basedir;
	this->dir_pose = "";
	this->dir_image = "%02d/";
	this->type_image = "%06d.png";
	this->type_pose = "%02d/pose.txt";
	this->left = "image_0/";
	this->right = "image_1/";
	this->calibfile = "calib.txt";
	this->timefile = "times.txt";	
}
void ImageDatabase::_readjustintrinsic(const double scfactor)
{
	this->Kleft = this->Kleft * scfactor;
	this->Kright = this->Kright * scfactor;
	this->Kleft(2, 2) = 1;
	this->Kright(2, 2) = 1;
}
// 