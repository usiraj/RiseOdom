/*
 * processodomdb.cpp
 *
 *  Created on: Mar 23, 2015
 *      Author: usama
 *  Process Odom Dataset offline
 */

#include "image_database.h"
#include "stereovodom.h"
#include "ispkf_motionmodels.h"
#include <stdio.h>

#include "stereopreprocess.h"

//some global variables
bool CmdLine_USELAB;
bool CmdLine_DISPLAY;
bool CmdLine_SPKF;
int CmdLine_SEQNO;
// some functions definitions
bool _parsecommandline(int argc, char **argv);
void _processdataset();

int main(int argc, char** argv){
	if (!_parsecommandline(argc, argv) ) return -1;
	_processdataset();
	return 0;
}
bool _parsecommandline(int argc, char** argv)
{
/*
	 * -s : sequence number
	 * -l : use lab datasets
	 * -h show help message
	 * -k : use spkf optimization
	 * -d : display images
	 */
	bool error = false;
	// default values
	CmdLine_SEQNO = 4;
	CmdLine_USELAB = false;
	CmdLine_DISPLAY = false;
	CmdLine_SPKF = false ;
	// check commands passed to program
	if ( argc > 1){
		for ( int i = 1; i < argc; ++i){
			if ( strcmp(argv[i],"-l") == 0){
				// Use Lab DataSet
				CmdLine_USELAB = true;
			} else if ( strcmp(argv[i], "-d") == 0 ) {
				// display
				CmdLine_DISPLAY = true;
			} else if ( strcmp(argv[i],"-s") == 0){
				// maximum retries
				if ( i+1 < argc ){
					++i;
					unsigned int pr = 0;
					sscanf(argv[i],"%hu",(short*)&pr);
					if (pr < 0 ){
						error = true;
						break;
					} else {
						// Set the sequence Number
						CmdLine_SEQNO = pr;
					}
				} else {
					error = true;
					break;
				}
			}  else if ( strcmp(argv[i],"-k") == 0){
				// use local optimization
				CmdLine_SPKF = true;
			} else {
				error = true;
				break;
			}
		}
	}
	if ( error ){
		printf("Usage : processodomdb [-s] [sequence number] [-l]\n");
		printf("\t\t use [-h] to show this message\n");
		printf("\t\t use [-k] to use local optimization\n");
		printf("\t\t [-l] will process lab dataset\n");
		printf("\t\t [-d] will display debugging video\n");
		fflush(stdout);
		return false;
	} else{
		return true;
	}
}
void _processdataset(){
	bool dosave = true;
	std::vector<TransformationMatrix> _trlist;
	cv::Mat imgl, imgr, imgo;
	double _frametime, _scaling = 1.;
	ImageDatabase *db;
	RiseOdom::StereoPreProcess impre;
	RiseOdom::StereoVodom sfun(3);
	ISPKF_2D ispkf(1E-2,2,0);
	ispkf.set_iteration_criteria(1E-6, 20);
	if ( CmdLine_USELAB ){
		db = new ImageDatabase(CmdLine_SEQNO);
		// set ISPKF_2D robot to camera transform
		TransformationMatrix bTcam;
		bTcam.setTranslation(cv::Matx31d(0.1525,0.02922,0.62995));
		bTcam.setRotationFixed(cv::Matx31d(-M_PI_2,0,-M_PI_2));
		ispkf.setBaseCameraTransform(bTcam);
		printf("Using Lab Dataset , sequence : %d\n", CmdLine_SEQNO);
	} else {
		db = new KittiDatabase(CmdLine_SEQNO);
		printf("Using Kitti Dataset , sequence : %d\n", CmdLine_SEQNO);
	}
	if ( CmdLine_SPKF ) {
		printf("spkf optimization will also be applied\n");
		TransformationMatrix bTcam;
		bTcam.setTranslation(cv::Matx31d(0.0,0.0,0.0));
		bTcam.setRotationFixed(cv::Matx31d(-M_PI_2,0,-M_PI_2));
		ispkf.setBaseCameraTransform(bTcam);
		sfun.EnableSPKF(true);
	} else {
		sfun.EnableSPKF(false);
	}	
	db->set_scale(_scaling);
	// initialize impre
	impre.initialize(240, 3000, 7, db->image_size());	
	impre.setCam(db->getIntrinsicLeft(), db->get_baseline(), 3.);
	impre.setMatching(3, 65, 0.75, 48);
	// initialize sfun
	sfun.initialize(db->get_baseline(), db->get_focus(), db->getIntrinsicLeft());
	sfun.setsearchcriteria(0.95, 48 , 90, true );
	sfun.setdesiredreprojectionerror(2.0);
	sfun.setitereations(0.995, 0.8);
	sfun.register_ispkf(&ispkf);
	// initializing ispkf
	ispkf.setstatecovs(1E-4, 1E-5, 1E-5, 1E-4, 1E-5);
	ispkf.setrmatcovs(1E-6, 1E-6, 1E-6, 0.05, 1E-6);
	ispkf.StartFilter();
	for ( int _i = 0;; ++_i) {
		printf("Frame %d\n", _i); fflush(stdout);
		if ( !(db->load_grayimage(_i, &imgl, &imgr, &_frametime))) break;
		//sfun.add_image(imgl, imgr, _frametime);
		impre.ProcessImages(imgl, imgr);
		RiseOdom::Frame_Data data = impre.getFrameData() ;data.frame_id = _i;
		sfun.add_imagedata(data, _frametime);
		if ( CmdLine_DISPLAY ){
			//imgo = RiseOdom::draw_framescanmatch(imgl, imgr, data);			/// DEBUG DRAW	
			imgo = imgl.clone();
			sfun.estimate_latest_motion(&imgo);
		} else sfun.estimate_latest_motion(NULL);
		sfun.local_optimization();
		_trlist.push_back(sfun.get_pose());
		if ( CmdLine_DISPLAY ){	
			cv::Point2f vel = sfun.get_vel();			
			imgo = RiseOdom::draw_transforminfo(imgo, sfun.get_pose(), vel.x, vel.y );
			if ( !imgo.empty() ){
				cv::imshow("match", imgo);
				int k = cv::waitKey(10) & 0xff;
				if ( k == ' ' ) k = cv::waitKey(0) & 0xff;
				if ( k == 27 ) break;
			}
		}
	}
	printf("Final Pose was");
	print_2dmat(cv::Mat(sfun.get_pose().getMatrix()));
	if ( dosave ){
		char flnametr[64],_algoname[64];
		char _algoName[] = "riseodom";
		char _pathtores[] = "Results";
		if ( CmdLine_SPKF ) {
			snprintf(_algoname, 64, "%s_spkf", _algoName);
		} else {
			snprintf(_algoname, 64, "%s", _algoName);
		}
		if ( CmdLine_USELAB ){
			snprintf(flnametr, 64, "%s/lab%02d_%spose.txt",_pathtores, CmdLine_SEQNO, _algoname);
		} else {
			snprintf(flnametr, 64, "%s/kitti%02d_%spose.txt", _pathtores, CmdLine_SEQNO, _algoname);
		}
		printf("Saving Pose to %s \n", flnametr);
		save_poses(flnametr, _trlist);
	}
	// clean
	delete db;
}
