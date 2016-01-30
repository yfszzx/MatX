template <typename TYPE, bool CUDA>
class searchTool{
private:
	//search相关
	enum Judge{OK = 0, LARGE = 1, SMALL = -1};
	MatGroup<TYPE, CUDA> lastGrad;
	MatGroup<TYPE, CUDA> direct;
	MatGroup<TYPE, CUDA> lastDirect;
	MatGroup<TYPE, CUDA> Pos;
	MatGroup<TYPE, CUDA> lastPos;
	double Loss;
	double Deriv;
	double Loss_t;
	double Deriv_t;
	double lastNorm;
	TYPE Step;
	TYPE maxPos;
	TYPE minPos;
	int moveCount;	
	bool startFlag;
	bool overFlag;
	bool lineOver;
	int algorithm;
	int count;
	inline double interpolation(double x,double v0,double v1,double derv0,double derv1);
	inline int wolfe_powell_judge();
	bool momentum_grad(MatGroup<TYPE, CUDA> &Ws,const MatGroup<TYPE, CUDA> &Grad, double loss);
	void init_pos(MatGroup<TYPE, CUDA> &Ws,const MatGroup<TYPE, CUDA> &Grad, double loss, bool searchOver = true);
	bool line_search(MatGroup<TYPE, CUDA> &Ws,const MatGroup<TYPE, CUDA> &Grad, double loss);
	void conjDirect(MatGroup<TYPE, CUDA> &Ws,const MatGroup<TYPE, CUDA> &Grad, bool searchOver);
	void LbfgsDirect(MatGroup<TYPE, CUDA> &Ws,const MatGroup<TYPE, CUDA> &Grad, bool searchOver);
	
	//LBFGS相关
	double *L_alf;
	double *L_ro;
	MatGroup<TYPE, CUDA> *L_s;
	MatGroup<TYPE, CUDA> *L_y;
	int L_num;
	int L_count;
	void L_free();
	void L_init(int num);
	
	//统计loss的变化的显著性(Z)
	int confirmRounds;
	int recorderCount;
	MatriX<float, false> lossRecorder;
	float lossMean;
	float lossMSE;
	void recordLoss(float loss);
	float Z;
	bool notableFlag;
public:
	
	//搜索参数
	enum Alg{ Momentum = 0, Fastest =1, Conjugate = 2, LBFGS = 3};
	double lr;
	TYPE momentum;
	double WP_value;
	double WP_deriv;
	TYPE initStep;	
	int maxMoveNum;
	float Zscale;
	bool debug;

	//返回值
	
	MatGroup<TYPE, CUDA> overPos;	//线搜结束后的最佳结果
	int rounds();
	void setRounds(int r);
	inline bool notable();
	inline float getZ();
	inline float getLoss();
	void reset();
	void setConfirmRounds(int n);
	void setAlg(int alg, int LbfgsNum = 5);
	void changeBatch();//更换batch时使用
	bool move(MatGroup<TYPE, CUDA> &Ws,const MatGroup<TYPE, CUDA> &Grad, double loss, bool randBatch = true);	
	searchTool();
	~searchTool();
};


