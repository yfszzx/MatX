template <typename TYPE, bool CUDA>
double searchTool<TYPE, CUDA>::interpolation(double x, double v0, double v1, double derv0, double derv1){//两点三次插值
	double s,z;
	double w;
	s = 3 * (v1 - v0) / x;
	z = s - derv1 - derv0;
	w = z * z - derv1 * derv0;
	if(w < 0){
		return -1;
	}
	w = sqrt(w);
	s = derv1 - derv0 + 2 * w;
	if(abs(s) < FLT_MIN){
		return -1;
	}
	x = x * (w - derv0 - z) / s;
	if(_finite(x)){
		return x;
	}
	return -1;
}
template <typename TYPE, bool CUDA>
int searchTool<TYPE, CUDA>::wolfe_powell_judge(){
	int ret = OK;
	if(Loss_t > Loss + WP_value * Step * Deriv){
		ret = LARGE;
	}else {
		if(Deriv_t < Deriv * WP_deriv){
			ret = SMALL;
		}
	}
	return ret;
}
template <typename TYPE, bool CUDA>
bool searchTool<TYPE, CUDA>::momentum_grad(MatXG &Ws,const MatXG &Grad, double loss){
	if(startFlag){
		lastGrad = Grad;
		lastGrad = 0;
		startFlag =false;
	}
	overPos = Ws;
	Ws -= TYPE(lr) * (Grad + lastGrad * momentum);
	lastGrad = Grad;
	count ++;
	if(debug){
		cout<<"\n"<<count<<"\tls"<<loss;
		getchar();
	}
	return true;
}
template <typename TYPE, bool CUDA>
void searchTool<TYPE, CUDA>::conjDirect(MatXG &Ws,const MatXG &Grad, bool searchOver){
	if(startFlag){
		lastNorm = Grad.norm();
		lastGrad = Grad;
		lastDirect = - Grad;
		Deriv = - lastNorm;
		direct = Grad/Deriv;
		startFlag = false;
		return;

	}
	double grad_norm = Grad.squaredNorm();
	double r=0;
	if(searchOver){
		r = ( grad_norm - lastGrad.dot(Grad))/lastNorm;

	}else{
		r = 0;		
	}
	lastNorm = grad_norm;
	lastGrad = Grad;
	lastDirect =  - Grad +  TYPE(r) * lastDirect ;
	direct  = lastDirect/lastDirect.norm();	
	Deriv = Grad.dot(direct);				

}
template <typename TYPE, bool CUDA>
void searchTool<TYPE, CUDA>::LbfgsDirect(MatXG &Ws,const MatXG &Grad, bool searchOver){
	if(startFlag || !searchOver){
		Deriv = - Grad.norm();
		direct = Grad/Deriv;
		startFlag = false;
		lastGrad = Grad;
		L_count = 0;
		return;
	}
	int l,ll;
	l=(L_count < L_num)?L_count:L_num;
	if(L_count < L_num){
		ll = L_count;
	}else{
		ll = L_num - 1;
		L_ro[ll] = L_ro[0];
		L_s[ll] = L_s[0];
		L_y[ll] = L_y[0];
		for(int i=0; i<ll; i++){
			L_ro[i] = L_ro[i + 1];
			L_s[i] = L_s[i + 1];
			L_y[i] = L_y[i + 1];					
		}
	}
	L_s[ll] = Ws - Pos;
	L_y[ll] = Grad - lastGrad;
	L_ro[ll] = L_s[ll].dot(L_y[ll]);
	MatXG tmp = Grad;
	for(int i = l - 1; i >= 0;i--){		
		L_alf[i] = L_s[i].dot(tmp)/L_ro[i];
		tmp -= TYPE(L_alf[i]) * L_y[i];
	}
	double bt;
	for(int i=0; i<l; i++){
		bt = L_y[i].dot(tmp)/L_ro[i];
		tmp +=  TYPE(L_alf[i] - bt) * L_s[i];
	}
	direct  = -tmp/tmp.norm();	
	Deriv = Grad.dot(direct);
	L_count ++;
	lastGrad = Grad;
}
template <typename TYPE, bool CUDA>
void searchTool<TYPE, CUDA>::init_pos(MatXG &Ws,const MatXG &Grad, double loss, bool searchOver = true){
	
	moveCount = 0;	
	if(!searchOver){
		cout<<"\t[interrupted]";	
		Step = avgStep ;
		
	}
	switch(algorithm){
	case Fastest:
		Deriv = - Grad.norm();
		direct = Grad/Deriv;
		break;				
	case Conjugate:
		conjDirect(Ws, Grad, searchOver);
		break;
	case LBFGS:
		LbfgsDirect(Ws, Grad, searchOver);	
		break;
	}
	Loss =loss;
	lineOver = true;
	Pos = Ws;
	overFlag = false;
	maxPos = 0;
	minPos = 0;
	int tmp = min(avgCount, 200);
	avgStep = (avgStep * tmp + Step)/(tmp + 1);
	avgCount ++;
	//Step = avgStep;
	
}
template <typename TYPE, bool CUDA>
bool searchTool<TYPE, CUDA>::line_search(MatXG &Ws,const MatXG &Grad, double loss){
	double x=0;
	if(overFlag){
		init_pos(Ws, Grad, loss);		
	}else{
		Deriv_t = direct.dot(Grad);
		Loss_t = loss;
		int jdg= wolfe_powell_judge();
		if(jdg == OK || moveCount > maxMoveNum){		
			init_pos(Ws, Grad, loss, jdg == OK );			
		}else{
			if(jdg == LARGE){
				maxPos = Step;
			}else{
				minPos = Step;
			}
			x = interpolation(Step, Loss, Loss_t, Deriv, Deriv_t);
			Step = x;
			if(x <= minPos || (maxPos > 0 && x >= maxPos)){
				if(maxPos == 0){
					Step = minPos * 2;
				}else{
					Step =(maxPos + minPos)/2;
				}
			}
		}
	}
	bool overFlag = (moveCount == 0);
	if(overFlag){
		overPos = Pos;
	};
	Ws = Pos + Step * direct;
	moveCount ++;
	count ++;
	if(debug){
		cout<<"\nls"<<Loss<<"\tdv"<<Deriv<<"\tlst"<<Loss_t<<"\tdvt"<<Deriv_t;
		cout<<"\nmc"<<moveCount<<"\tst"<<Step<<"\tmx"<<maxPos<<"\tmn"<<minPos<<"\tx"<<x;
		cout<<"\navgStep"<<avgStep<<"\tavgCount"<<avgCount;
		getchar();
	}
	return  overFlag;
}
template <typename TYPE, bool CUDA>
searchTool<TYPE, CUDA>::searchTool(){
	algorithm = 0;
	lr = 1;
	momentum = 0.8;
	WP_value=0.4;
	WP_deriv=0.1;
	maxMoveNum = 5;
	debug = false;
	L_alf = NULL;
	L_num = 5;
	initStep = 0.1;
	confirmRounds = 10;
	reset();
}
template <typename TYPE, bool CUDA>
searchTool<TYPE, CUDA>::~searchTool(){
	L_free();
}
template <typename TYPE, bool CUDA>
void searchTool<TYPE, CUDA>::setConfirmRounds(int n){
	confirmRounds = n;
	lossRecorder =  MatriX<float, false>::Zero(confirmRounds, 1);
}
template <typename TYPE, bool CUDA>
void searchTool<TYPE, CUDA>::recordLoss(float loss){
	lossRecorder.assignment(recorderCount, loss);
	recorderCount ++;
	if(recorderCount == confirmRounds){
		float currentMean;
		float currentMSE = lossRecorder.allMSE(currentMean);
		if(lossMean == -1){
			lossMean = currentMean;
			lossMSE = currentMSE;
			//Z =currentMean/sqrt(currentMSE/confirmRounds);
		}else{
			if(currentMSE + lossMSE < FLT_MIN){
				Z = -1000;
			}else{
				Z =(lossMean - currentMean)/sqrt((currentMSE + lossMSE)/confirmRounds);
			}
			if(Z < Zscale ){
				notableFlag = false;
			}
			cout<<"\nmean:"<<lossMean<<" "<<currentMean<<"\tMSE:"<<lossMSE<<" "<<currentMSE<<"\tZ:"<<Z;
			lossMean = -1;
			lossMSE = -1;
		}
		
		recorderCount = 0;

	}
};
template <typename TYPE, bool CUDA>
int searchTool<TYPE, CUDA>::rounds(){
	return count;
}
template <typename TYPE, bool CUDA>
void searchTool<TYPE, CUDA>::setRounds(int r){
	count = r;
}
template <typename TYPE, bool CUDA>
bool searchTool<TYPE, CUDA>::notable(){
	return notableFlag;
};
template <typename TYPE, bool CUDA>
bool searchTool<TYPE, CUDA>::move(MatXG &Ws,const MatXG &Grad, double loss, bool randBatch){
	notableFlag = true;
	if(randBatch){
		if(startFlag){
			recordLoss(loss);
		}
	}else{
		if(lineOver){
			recordLoss(Loss);
			lineOver = false;
		}

	}
	if(algorithm == 0){
		lineOver = true;
		return momentum_grad(Ws, Grad, loss);
	}else{
		return line_search(Ws, Grad, loss);
	}

}
template <typename TYPE, bool CUDA>
void searchTool<TYPE, CUDA>::changeBatch(){
	startFlag = true;
	overFlag = true;
	lineOver = false;
}
template <typename TYPE, bool CUDA>
void searchTool<TYPE, CUDA>::setAlg(int alg, int LbfgsNum){
	algorithm = alg;
	if(algorithm == 3){
		L_init(LbfgsNum);
	}else{
		L_free();
	}
}
template <typename TYPE, bool CUDA>
float searchTool<TYPE, CUDA>::getZ(){
	return Z;
}
template <typename TYPE, bool CUDA>
float searchTool<TYPE, CUDA>::getLoss(){
	return Loss;
}
template <typename TYPE, bool CUDA>
void searchTool<TYPE, CUDA>::L_free(){
	if(L_alf != NULL){
		delete [] L_alf;			
		delete [] L_ro;
		delete [] L_s;
		delete [] L_y;
	}
	L_alf = NULL;
};
template <typename TYPE, bool CUDA>
void searchTool<TYPE, CUDA>::L_init(int num){

	L_free();
	L_num = num;
	L_alf = new double[L_num];
	L_ro = new double[L_num];
	memset(L_alf, 0, _msize(L_alf));
	memset(L_ro, 0, _msize(L_ro));
	L_s =new MatXG[L_num];
	L_y =new MatXG[L_num];
	changeBatch();
	L_count = 0;
};
template <typename TYPE, bool CUDA>
void searchTool<TYPE, CUDA>::reset(){
	Step = initStep;
	avgStep = Step;
	avgCount = 1;
	count = 0;	
	recorderCount = 0;
	Z = 1000;
	setConfirmRounds(confirmRounds);
	setAlg(algorithm, L_num);
	lossMean = -1;
	lossMSE = -1;	
	changeBatch();
	notableFlag = true;
	
};