/*
 * Copyright 2015 <Usama Siraj> <osama@smme.edu.pk>
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
 *  Implements UKF variation of SPKF and ISPKF with weights of UKF
 */

#ifndef SPKF_H
#define SPKF_H

#include <opencv2/opencv.hpp>

typedef cv::Mat (*TransitionFunc)(const cv::Mat &input, const double _delt, void *ptrdata, const cv::Mat noise);	// input is a nx1 matrix
typedef cv::Mat (*ObservationFunc)(const cv::Mat &input, void *ptrdata, const cv::Mat noise);

class SPKF
{
public:
	SPKF(const unsigned int dimX, const double _alpha = 1E-4, const double _beta = 2, const double _k = 0);
	SPKF(const SPKF &other);
	virtual ~SPKF();
	// initialization
	void set_iteration_criteria(const double convergence, const double max_iter);
	void setQ(const cv::Mat &Q);
	void setQDiagonal(const cv::Mat &diag);
	void setX(const cv::Mat &X);
	// register transtion functions
	void register_transition_func(TransitionFunc func);					// set it to null to use linear prediction
	int register_obsrvfunc(ObservationFunc func, const cv::Mat &R);		// returns idx of observation function
	void unregister_obsrvfunc(const int idx);
	void unregister_obsrvall();
	// get members
	cv::Mat getP() const;
	cv::Mat getX() const;
	cv::Mat getQ() const;
	int _getLx() const;
	// start filter
	virtual void StartFilter();											// inherited classes may add custom functionality to it
	void Start();
	void SigmaPrediction(const double delt, void *ptr = NULL);
	void MeasurementUpdate(const cv::Mat &z, const int idx, void *ptrdata = NULL, bool iterative = false);
	// some helping static functions
	static cv::Mat matrix_squareroot(const cv::Mat &square_matrix);
	static double calculate_lambda(const unsigned int L, const double alpha = 1E-4, const double k = 0);
	static cv::Mat calculate_weights_mean(const unsigned int L, const double lambda);
	static cv::Mat calculate_weights_covariance(const unsigned int L, const double lambda, const double alpha, const double beta);
	static cv::Mat calculate_Zmean(const cv::Mat &Z, const cv::Mat &wtsm);
	static cv::Mat calculate_covZ(const cv::Mat &Z, const cv::Mat  &Zmean, const cv::Mat &wtsc);
	static cv::Mat calculate_covXZ(const cv::Mat &Z, const cv::Mat &Zmean, const cv::Mat &sig_pts,
							   const cv::Mat &xmean, const cv::Mat &wtsc);
	static cv::Mat calculate_sigmamat(const cv::Mat &P, const double lambda, const int L);
	static cv::Mat calculate_sigmapts(const cv::Mat &xmean, const cv::Mat &sigmamat);
protected:
	// for iterative version
	double _convergence;
	double _max_iter;
	//
	unsigned int Lx;		// augmented dimension
	double alpha;
	double beta;
	double k;
	// states and covariances
	cv::Mat _Q;
	// augmented state and covariance ; process them after Start is called
	cv::Mat _X;
	cv::Mat _P;
	// for measurements
	TransitionFunc _statetransition;
	std::vector<ObservationFunc> _obsrvfunc;
	std::vector<cv::Mat> _R;
	cv::Mat _batchcorrect_constraints(const cv::Mat &mat) const;
	virtual cv::Mat _correct_constraints(const cv::Mat &mat) const;	
private:
	// core functions
	void _sigma_prediction(const double delt, void *ptr);
	void _update_measurement_ukf(const cv::Mat &z, const int idx, void *ptr);
	void _update_measurement_ispkf(const cv::Mat &z, const int idx, void *ptr);
	// helping functions
	cv::Mat _cvectorapply_predictionfunc(const cv::Mat &xcvec,const double delt, void *ptr, const cv::Mat &noise) const;
	cv::Mat _cvectorapply_observationfunc(const cv::Mat &xcvec, const int idx, void *ptr, const cv::Mat &noise) const;
};


#endif // SPKF_H
