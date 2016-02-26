template <typename TYPE, bool CUDA>
class searchTool{
private:
	//search���
	enum Judge{OK = 0, LARGE = 1, SMALL = -1};
	MatXG lastGrad;
	MatXG direct;
	MatXG lastDirect;
	MatXG Pos;
	MatXG lastPos;
	double Loss;
	double Deriv;
	double Loss_t;
	double Deriv_t;
	double lastNorm;
	TYPE Step;
	TYPE maxPos;
	TYPE minPos;

	int avgCount;
	int moveCount;	
	bool startFlag;
	bool overFlag;
	bool lineOver;
	int algorithm;
	int count;
	//float maxStepScale;
	inline double interpolation(double x,double v0,double v1,double derv0,double derv1);
	inline int wolfe_powell_judge();
	bool momentum_grad(MatXG &Ws,const MatXG &Grad, double loss);
	void init_pos(MatXG &Ws,const MatXG &Grad, double loss, bool searchOver = true);
	bool line_search(MatXG &Ws,const MatXG &Grad, double loss);
	void conjDirect(MatXG &Ws,const MatXG &Grad, bool searchOver);
	void LbfgsDirect(MatXG &Ws,const MatXG &Grad, bool searchOver);
	
	//LBFGS���
	double *L_alf;
	double *L_ro;
	MatXG *L_s;
	MatXG *L_y;
	int L_num;
	int L_count;
	void L_free();
	void L_init(int num);
	
	//ͳ��loss�ı仯��������(Z)
	int confirmRounds;
	int recorderCount;
	MatCF lossRecorder;
	float lossMean;
	float lossMSE;
	void recordLoss(float loss);
	float Z;
	bool notableFlag;
	float maxStep;
public:
	
	//��������
	enum Alg{ Momentum = 0, Fastest =1, Conjugate = 2, LBFGS = 3};
	double lr;
	TYPE momentum;
	double WP_value;
	double WP_deriv;
	TYPE initStep;	
	int maxMoveNum;
	float Zscale;
	bool debug;
	double avgStep;

	//����ֵ
	
	MatXG overPos;	//���ѽ��������ѽ��
	int rounds();
	void setRounds(int r);
	inline bool notable();
	inline float getZ();
	inline float getLoss();
	void reset();
	void setConfirmRounds(int n);
	void setAlg(int alg, int LbfgsNum = 5);
	void changeBatch();//����batchʱʹ��
	bool move(MatXG &Ws,const MatXG &Grad, double loss, bool randBatch = true);	
	searchTool();
	~searchTool();
};


