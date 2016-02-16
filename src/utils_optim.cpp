/*
 * utils_optim.cpp
 *
 *  Created on: March 11, 2015
 *      Author: usama
 */
#include "utils_optim.h"
#include "riseodomsettings.h"
#include <stdio.h>

#ifdef USE_PENTIUM4
int _mm_popcnt_u32(const unsigned int _num){
    int cnt = 0;
    unsigned int num = _num;
    for ( int _i = 0; _i < 4; ++_i ){
        if ( num & 0x00000001 ){
            cnt += 1;       
        }
        if ( num & 0x00000002 ){
            cnt += 1;
        }
        if ( num & 0x00000004 ){
            cnt += 1;       
        }
        if ( num & 0x00000008 ){
            cnt += 1;
        }
        if ( num & 0x00000010 ){
            cnt += 1;       
        }
        if ( num & 0x00000020 ){
            cnt += 1;
        }
        if ( num & 0x00000040 ){
            cnt += 1;       
        }
        if ( num & 0x00000080 ){
            cnt += 1;
        }
        num >>= 8;
    }
    return cnt;
}
#endif
int _normoptimized(const unsigned char* a, const unsigned char* b, const int n)
{
#ifdef USE_PENTIUM4
	int _ddt = 0;
	unsigned int _dis, _dis2, _dis3, _dis4;
	long int _cnt, _cnt2, _cnt3, _cnt4;
	for (int _i = 0; _i < n; _i+=16){
		// xor 128 bits
		__m128i a0 = _mm_loadu_si128((const __m128i*)(a + _i));
		__m128i b0 = _mm_loadu_si128((const __m128i*)(b + _i));
		b0 = _mm_xor_si128(a0, b0);
		__m128i d = _mm_srli_si128(b0, 4);
	    __m128i e = _mm_srli_si128(d,4);
	    __m128i f = _mm_srli_si128(e,4);
	    _dis = _mm_cvtsi128_si32(b0);
	    _dis2 = _mm_cvtsi128_si32(d);
	    _dis3 = _mm_cvtsi128_si32(e);
	    _dis4 = _mm_cvtsi128_si32(f);
	    // now count
	    _cnt = _mm_popcnt_u32(_dis);
	    _cnt2 = _mm_popcnt_u32(_dis2);
	    _cnt3 = _mm_popcnt_u32(_dis3);
	    _cnt4 = _mm_popcnt_u32(_dis4);
	    _ddt += _cnt + _cnt2 + _cnt3 + _cnt4;
	}
	return _ddt;
#else	
	int _ddt = 0;
	unsigned long int _dis, _dis2;
	long int _cnt, _cnt2;
	for (int _i = 0; _i < n; _i+=16){
		// xor 128 bits
		__m128i a0 = _mm_loadu_si128((const __m128i*)(a + _i));
		__m128i b0 = _mm_loadu_si128((const __m128i*)(b + _i));
		b0 = _mm_xor_si128(a0, b0);
		a0 = _mm_srli_si128(b0,8);
		_dis = _mm_cvtsi128_si64(b0);
		_dis2 = _mm_cvtsi128_si64(a0);
		_cnt = _mm_popcnt_u64(_dis);
		_cnt2 = _mm_popcnt_u64(_dis2);
		_ddt +=  _cnt + _cnt2;		// other commmands don't give any advantage
	}
	return _ddt;
#endif
}
int _normoptimized64(const unsigned char*a, const unsigned char*b){
#ifdef USE_PENTIUM4
	return _normoptimized(a,b,64);
#else	
	unsigned long int _dis0a, _dis0b, _dis1a, _dis1b, _dis2a, _dis2b, _dis3a, _dis3b;
	long int _cnt0a, _cnt0b, _cnt1a, _cnt1b, _cnt2a, _cnt2b, _cnt3a, _cnt3b;
	// first
	__m128i a0 = _mm_loadu_si128((const __m128i*)(a));
	__m128i a1 = _mm_loadu_si128((const __m128i*)(a+16));
	__m128i a2 = _mm_loadu_si128((const __m128i*)(a+32));
	__m128i a3 = _mm_loadu_si128((const __m128i*)(a+48));
	__m128i b0 = _mm_loadu_si128((const __m128i*)(b));
	__m128i b1 = _mm_loadu_si128((const __m128i*)(b+16));
	__m128i b2 = _mm_loadu_si128((const __m128i*)(b+32));
	__m128i b3 = _mm_loadu_si128((const __m128i*)(b+48));
	b0 = _mm_xor_si128(a0, b0);
	b1 = _mm_xor_si128(a1, b1);
	b2 = _mm_xor_si128(a2, b2);
	b3 = _mm_xor_si128(a3, b3);
	a0 = _mm_srli_si128(b0,8);
	a1 = _mm_srli_si128(b1,8);
	a2 = _mm_srli_si128(b2,8);
	a3 = _mm_srli_si128(b3,8);
	_dis0a = _mm_cvtsi128_si64(b0);
	_dis0b = _mm_cvtsi128_si64(a0);
	_dis1a = _mm_cvtsi128_si64(b1);
	_dis1b = _mm_cvtsi128_si64(a1);
	_dis2a = _mm_cvtsi128_si64(b2);
	_dis2b = _mm_cvtsi128_si64(a2);
	_dis3a = _mm_cvtsi128_si64(b3);
	_dis3b = _mm_cvtsi128_si64(a3);
	_cnt0a = _mm_popcnt_u64(_dis0a);
	_cnt0b = _mm_popcnt_u64(_dis0b);
	_cnt1a = _mm_popcnt_u64(_dis1a);
	_cnt1b = _mm_popcnt_u64(_dis1b);
	_cnt2a = _mm_popcnt_u64(_dis2a);
	_cnt2b = _mm_popcnt_u64(_dis2b);
	_cnt3a = _mm_popcnt_u64(_dis3a);
	_cnt3b = _mm_popcnt_u64(_dis3b);
	return _cnt0a + _cnt0b + _cnt1a + _cnt1b + _cnt2a + _cnt2b + _cnt3a + _cnt3b;
#endif
}
int _normoptimized48(const unsigned char* a, const unsigned char* b)
{
#ifdef USE_PENTIUM4
	return _normoptimized(a,b,48);
#else
	unsigned long int _dis0a, _dis0b, _dis1a, _dis1b, _dis2a, _dis2b;
	long int _cnt0a, _cnt0b, _cnt1a, _cnt1b, _cnt2a, _cnt2b;
	// first
	__m128i a0 = _mm_loadu_si128((const __m128i*)(a));
	__m128i a1 = _mm_loadu_si128((const __m128i*)(a+16));
	__m128i a2 = _mm_loadu_si128((const __m128i*)(a+32));
	__m128i b0 = _mm_loadu_si128((const __m128i*)(b));
	__m128i b1 = _mm_loadu_si128((const __m128i*)(b+16));
	__m128i b2 = _mm_loadu_si128((const __m128i*)(b+32));
	b0 = _mm_xor_si128(a0, b0);
	b1 = _mm_xor_si128(a1, b1);
	b2 = _mm_xor_si128(a2, b2);
	a0 = _mm_srli_si128(b0,8);
	a1 = _mm_srli_si128(b1,8);
	a2 = _mm_srli_si128(b2,8);
	_dis0a = _mm_cvtsi128_si64(b0);
	_dis0b = _mm_cvtsi128_si64(a0);
	_dis1a = _mm_cvtsi128_si64(b1);
	_dis1b = _mm_cvtsi128_si64(a1);
	_dis2a = _mm_cvtsi128_si64(b2);
	_dis2b = _mm_cvtsi128_si64(a2);
	_cnt0a = _mm_popcnt_u64(_dis0a);
	_cnt0b = _mm_popcnt_u64(_dis0b);
	_cnt1a = _mm_popcnt_u64(_dis1a);
	_cnt1b = _mm_popcnt_u64(_dis1b);
	_cnt2a = _mm_popcnt_u64(_dis2a);
	_cnt2b = _mm_popcnt_u64(_dis2b);
	return _cnt0a + _cnt0b + _cnt1a + _cnt1b + _cnt2a + _cnt2b;
#endif
}
int _normoptimized32(const unsigned char* a, const unsigned char* b)
{
#ifdef USE_PENTIUM4
	return _normoptimized(a,b,32);
#else
	unsigned long int _dis0a, _dis0b, _dis1a, _dis1b;
	long int _cnt0a, _cnt0b, _cnt1a, _cnt1b;
	// first
	__m128i a0 = _mm_loadu_si128((const __m128i*)(a));
	__m128i a1 = _mm_loadu_si128((const __m128i*)(a+16));
	__m128i b0 = _mm_loadu_si128((const __m128i*)(b));
	__m128i b1 = _mm_loadu_si128((const __m128i*)(b+16));
	b0 = _mm_xor_si128(a0, b0);
	b1 = _mm_xor_si128(a1, b1);
	a0 = _mm_srli_si128(b0,8);
	a1 = _mm_srli_si128(b1,8);
	_dis0a = _mm_cvtsi128_si64(b0);
	_dis0b = _mm_cvtsi128_si64(a0);
	_dis1a = _mm_cvtsi128_si64(b1);
	_dis1b = _mm_cvtsi128_si64(a1);
	_cnt0a = _mm_popcnt_u64(_dis0a);
	_cnt0b = _mm_popcnt_u64(_dis0b);
	_cnt1a = _mm_popcnt_u64(_dis1a);
	_cnt1b = _mm_popcnt_u64(_dis1b);
	return _cnt0a + _cnt0b + _cnt1a + _cnt1b;
#endif
}
int _normoptimized16(const unsigned char* a, const unsigned char* b)
{
#ifdef USE_PENTIUM4
	return _normoptimized(a,b,16);
#else
	unsigned long int _dis, _dis2;
	long int _cnt, _cnt2;
	__m128i a0 = _mm_loadu_si128((const __m128i*)(a));
	__m128i b0 = _mm_loadu_si128((const __m128i*)(b));
	__m128i c = _mm_xor_si128(a0, b0);
	__m128i d = _mm_srli_si128(c,8);
	_dis = _mm_cvtsi128_si64(c);
	_dis2 = _mm_cvtsi128_si64(d);
	_cnt = _mm_popcnt_u64(_dis);
	_cnt2 = _mm_popcnt_u64(_dis2);
	int _ddt =  _cnt + _cnt2;		// other commmands don't give any advantage
	return _ddt;
#endif
}
int _bestmatch64(const unsigned char *vec1, const cv::Mat& des2, int &dist, const float ratio, const int _par=20,
	const unsigned char *neightest=NULL)
{
	int _idx=-1; int _mindist1 = 10*_par;		int _mindist2 = 1000000;	// some arbirtrary large value for initializing
	// compute best idx for row _i of _des1
	int _n2 = des2.rows;
	int _val;
	int step = des2.step / sizeof(des2.ptr()[0]);
	for ( int _j = 0; _j < _n2; ++_j ){
		if ( neightest != NULL && neightest[_j] == 0 ){
			continue;		// skipping due to failed neighborhood test
		}
		// calculate hamming distance
		_val = _normoptimized64(vec1, des2.data+step*_j);
		if ( _val > _par) {continue;}//  must pass par score
		//_val = _normoptimized48(vec1+16, des2.data+step*_j+16);	// opencv does'nt implement cascade freak
		if ( _val < _mindist2 ){
		if ( _val < _mindist1 ){
			_mindist2 = _mindist1;
			_mindist1 = _val;
			_idx = _j;
		} else _mindist2 = _val;
		}
	}
	// ratio test
	if (_mindist1 < int(float(_mindist2) * ratio )){
		dist = _mindist1;
		return _idx;
	}
	return -1;
}
int _bestmatch(const unsigned char *vec1, const cv::Mat &des2, int &dist, const float ratio, const int par = 20,
			   const unsigned char *neightest=NULL){
	int _idx=-1; int _mindist1 = 1000;	int _mindist2 = 1000000;	// some arbirtrary large value for initializing
	// compute best idx for row _i of _des1
	int _n2 = des2.rows;
	int step = des2.step / sizeof(des2.ptr()[0]);
	for ( int _j = 0; _j < _n2; ++_j ){
		if ( neightest != NULL && neightest[_j] == 0 ){
			continue;		// skipping due to failed neighborhood test
		}
		// calculate hamming distance
		int _val = _normoptimized(vec1, des2.data+step*_j, des2.cols);
		if ( _val > par) {continue;}//  must pass par score
		if ( _val < _mindist2 ){
		if ( _val < _mindist1 ){
			_mindist2 = _mindist1;
			_mindist1 = _val;
			_idx = _j;
		} else _mindist2 = _val;
		}
	}
	// ratio test
	if (_mindist1 < int(float(_mindist2) * ratio )){
		dist = _mindist1;
		return _idx;
	}
	return -1;
}
/*
 * Opencv Parallel for 
 */
class HammingMatchBatch : public cv::ParallelLoopBody
{
public:
	HammingMatchBatch(const HammingMatchBatch &cpy);
    HammingMatchBatch(const cv::Mat &_des1,const cv::Mat &_des2, const cv::Mat &neighborhood ,const float _ratio=0.45,
		const int parscore = FASTHAMMING_FREAK_PARSCORE	)
    {
	desA = _des1;
	desB = _des2;
	neighbor = neighborhood;
	this->par_score = parscore;
	this->ratio = _ratio;
	_n1 = _des1.rows;
	_n2 = _des2.rows;
	step = _des1.step / sizeof(_des1.ptr()[0]);
	if ( neighbor.empty() ){
		stepn = -1;
	} else {
		stepn = neighbor.step / sizeof(neighbor.ptr()[0]);
	}
	result = cv::Mat::zeros(_n1, 3, CV_32S);
	for ( int _i = 0; _i < _n1; ++_i ){
		this->result.at<int>(_i, 0) = -1;
	}
    }
	void getResult(MatchList &good) const;
    void operator()(const cv::Range& range) const
    {
		if ( this->stepn != -1 ){
			for( int i = range.start; i < range.end; ++i )
			{
				this->_trybestidxsinglerow(this->desA.data+this->step*i, i, this->neighbor.data+this->stepn*i);
			}
		} else {
			for( int i = range.start; i < range.end; ++i )
			{
				this->_trybestidxsinglerow(this->desA.data+this->step*i, i, NULL);
			}
		}
    }
	virtual ~HammingMatchBatch(){}
protected:
	cv::Mat desA;
	cv::Mat desB;
	cv::Mat neighbor;
	cv::Mat result;
	float ratio;
	int par_score;
	int _n1, _n2;
	int step, stepn;
	void _trybestidxsinglerow(const unsigned char *veca, const int _i, const unsigned char *nghb) const;
};
void HammingMatchBatch::_trybestidxsinglerow(const unsigned char *_rowA, const int _i, const unsigned char *nghb) const{
	cv::Mat *_res = const_cast<cv::Mat*>(&(this->result));
	int dist, idx;
	if ( this->desB.cols == 64 ){
		idx = _bestmatch64(_rowA, this->desB, dist, this->ratio, this->par_score, nghb);
	} else {
		idx = _bestmatch(_rowA, this->desB, dist, this->ratio, this->par_score, nghb);
	}
	if ( idx != -1 ){
		_res->at<int>(_i, 0) = _i;	// the first
		_res->at<int>(_i, 1) = idx;	// the second
		_res->at<int>(_i, 2) = dist;
	}
}
HammingMatchBatch::HammingMatchBatch(const HammingMatchBatch &cpy){
	this->par_score = cpy.par_score;
	this->desA = cpy.desA;
	this->desB = cpy.desB;
	this->neighbor = cpy.neighbor;
	this->ratio = cpy.ratio;
	this->result = cpy.result;
	this->_n1 = cpy._n1;
	this->_n2 = cpy._n2;
	this->step = cpy.step;
	this->stepn = cpy.stepn;
}
void HammingMatchBatch::getResult(MatchList &good) const{
	good.clear();
	cv::DMatch _mtch;
	for ( int _i = 0; _i < this->result.rows; ++_i){
		_mtch.queryIdx = this->result.at<int>(_i, 0);
		_mtch.trainIdx = this->result.at<int>(_i, 1);
		_mtch.distance = this->result.at<int>(_i, 2);
		if ( _mtch.queryIdx != -1 ) good.push_back(_mtch);
	}
}
// some functions
int fast_hammingnorm(const unsigned char* a, const unsigned char* b, const int n)
{
	return _normoptimized(a, b, n);
}
void _hammingmatch_lowetest_slow(const cv::Mat &_des1, const cv::Mat &_des2, MatchList &_good, const float ratio){
	// compute and copmare norms row by row
	int _n1 = _des1.rows;
	int _n2 = _des2.rows;
	int _idx, _mindist1, _mindist2;
	cv::DMatch _mtch;
	for ( int _i = 0; _i < _n1; ++_i ){
		_idx=-1;	_mindist1 = 10000;	_mindist2 = 1000000;	// some arbirtrary large value for initializing
		// compute best idx for row _i of _des1
		for ( int _j = 0; _j < _n2; ++_j ){
			// calculate hamming distance
			int _val = cv::normHamming(_des1.row(_i).data, _des2.row(_j).data, 4);
			if ( _val > 8) {continue;}	// skip to make it fast
			_val = cv::normHamming(_des1.row(_i).data, _des2.row(_j).data, _des2.cols);
			if ( _val < _mindist2 ){
			if ( _val < _mindist1 ){
				_mindist2 = _mindist1;
				_mindist1 = _val;
				_idx = _j;
			} else _mindist2 = _val;
			}
		}
		// ratio test
		if ( (_idx != -1) && (_mindist1 < (_mindist2 * ratio ))){
			// all is okay
			printf("distance %d: %d\n", _i, _mindist1);
			_mtch.distance = _mindist1;
			_mtch.queryIdx = _i;		// the first
			_mtch.trainIdx = _idx;		// the second
			_good.push_back(_mtch);
		}
	}
}

void _hammingmatchserial(const cv::Mat &des1, const cv::Mat &des2, MatchList &good, const float ratio){
	int _n1 = des1.rows;
	int dist, idx;
	cv::DMatch _mtch;
	int step = des1.step / sizeof(des1.ptr()[0]);
	for (int _i = 0; _i < _n1; ++_i){
		if ( des2.cols == 64 ){
			idx = _bestmatch64(des1.data+step*_i, des2, dist, ratio, FASTHAMMING_FREAK_PARSCORE);
		} else {
			idx = _bestmatch64(des1.data+step*_i, des2, dist, ratio);
		}
		if ( idx != -1){
			_mtch.distance = dist;
			_mtch.queryIdx = _i;
			_mtch.trainIdx = idx;
			good.push_back(_mtch);
		}
	}
}

void fast_hammingnorm_matchserial(const cv::Mat &des1, const cv::Mat &des2, ListIdxCorrespondance &_list, const float ratio){
	MatchList good;
	good.clear();
	bool flipcorrespondence = false;
	if ( des1.rows <= des2.rows ){
		_hammingmatchserial(des1, des2, good, ratio);
	} else {
		flipcorrespondence = true;
		_hammingmatchserial(des2, des1, good, ratio);
	}	
	correspondance_list(good, _list, flipcorrespondence);
	
	
}

void fast_hammingnorm_match(const cv::Mat &des1, const cv::Mat&des2, ListIdxCorrespondance &_list,
							const float ratio, const int parscore){
	// Fast Hamming Norm based test
	cv::setNumThreads(3);
	cv::Mat empty;
	MatchList good;
	good.clear();
	bool flipcorrespondence = false;
	if ( des1.rows <= des2.rows ){
		HammingMatchBatch _hmtbb(des1, des2, empty, ratio, parscore);		
		cv::parallel_for_(cv::Range(0, des1.rows), _hmtbb, 3);
		_hmtbb.getResult(good);
	} else {
		flipcorrespondence = true;
		HammingMatchBatch _hmtbb(des2, des1, empty, ratio, parscore);
		cv::parallel_for_(cv::Range(0, des2.rows), _hmtbb, 3);
		_hmtbb.getResult(good);
	}	
	correspondance_list(good, _list, flipcorrespondence);
}
void fast_hammingnorm_match_neighborhood(const cv::Mat& des1, const cv::Mat& des2, const cv::Mat& neighborhood,
										 ListIdxCorrespondance& _list, const float ratio, const int parscore)
{
	cv::setNumThreads(3);
	MatchList good;
	good.clear();
	bool flipcorrespondence = false;
	if ( des1.rows <= des2.rows ){
		HammingMatchBatch _hmtbb(des1, des2,neighborhood, ratio, parscore);
		cv::parallel_for_(cv::Range(0, des1.rows), _hmtbb, 3);
		_hmtbb.getResult(good);
	} else {
		flipcorrespondence = true;
		cv::Mat _neight=neighborhood.t();
		HammingMatchBatch _hmtbb(des2, des1, _neight, ratio, parscore);
		cv::parallel_for_(cv::Range(0, des2.rows), _hmtbb, 3);
		_hmtbb.getResult(good);
	}	
	correspondance_list(good, _list, flipcorrespondence);
}
