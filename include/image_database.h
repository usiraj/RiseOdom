/*
 * KittiDatabase.h
 *
 *  Created on: Feb 5, 2015
 *      Author: usama
 */

#ifndef KITTIDATABASE_H_
#define KITTIDATABASE_H_

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#define _KITTIODOM_DEFAULT_DIRECTORY_BASE "/mnt/work/Research/DataBases/Vision/Kitti/dataset/"
#define _LABODOM_DEFAULT_DIRECTORY_BASE "/mnt/work/Research/DataBases/Vision/labdatabase/"

class ImageDatabase {
public:
	ImageDatabase();
	ImageDatabase(const unsigned int dataset, const char* basedir=_LABODOM_DEFAULT_DIRECTORY_BASE);
	virtual ~ImageDatabase();
	// some helper functions
	static bool isImageFileEmpty(const std::vector<std::string>& _name);
	// member functions
	cv::Matx33d getIntrinsicLeft() const;
	cv::Matx33d getIntrinsicRight() const;
	cv::Matx34d getExtrinsicLeft() const;
	cv::Matx34d getExtrinsicRight() const;
	cv::Matx34d getGroundTruth(const int _seq) const;
	double get_baseline() const;
	double get_focus() const;
	std::vector<std::string> get_image(const unsigned int _seq, double *timing=0) const;
	bool load_image(const unsigned int _seq, cv::Mat *imleft, cv::Mat *imright = 0, double *timing=0) const;
	bool load_grayimage(const unsigned int _seq, cv::Mat *imleft, cv::Mat *imright=0, double *timing=0) const;
	cv::Size image_size() const;
	bool is_groundtruth_available() const;
	void set_scale(const double scale);
	double get_scale() const;
protected:
	std::string dir_base;
	unsigned int dataset;
	std::vector<float> timings;
	std::vector<cv::Matx34d> ground_truth;
	cv::Matx33d Kleft, Kright;
	cv::Matx34d Pleft, Pright;
	double scalingfactor;
	// some other members
	std::string dir_pose;
	std::string dir_image;
	std::string type_image;
	std::string type_pose;
	std::string left;
	std::string right;
	std::string calibfile;
	std::string timefile;
	// utility functions
	virtual void _build_base_db(const char* _basedir);
	void _load_timings();
	virtual void _load_calibration();
	virtual bool _dbnum_exist() const;
	virtual bool _pose_exist() const;
	virtual bool _imgseq_exist(const unsigned int _seq) const;
	std::vector<std::string> _getfile_image(const unsigned int_seq) const;
	std::string _getfile_callib() const;
	std::string _getfile_time() const;
	std::string _getfile_pose() const;
	void _readjustintrinsic(const double scfactor);
};


// Derived Class Kitti Database : for accessing KittiDatabase



class KittiDatabase : public ImageDatabase {
public:
	KittiDatabase(const unsigned int dataset = 4, const char* basedir=_KITTIODOM_DEFAULT_DIRECTORY_BASE);
	virtual ~KittiDatabase();
protected:
	virtual void _build_base_db(const char* _basedir);
	virtual void _load_calibration();
	virtual bool _dbnum_exist() const;
	virtual bool _pose_exist() const;
	virtual bool _imgseq_exist(const unsigned int _seq) const;
private:
	std::vector<unsigned int> poses;
	std::vector<unsigned int> sequences;
	std::vector<unsigned int> lastimage;
};

std::vector<cv::Matx34d> load_poses(const char* filename);
void save_poses(const char *filename,const std::vector<cv::Matx34d> &_poses);
#endif /* KITTIDATABASE_H_ */
