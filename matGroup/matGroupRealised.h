template <typename TYPE, bool CUDA >
int MatGroup<TYPE, CUDA>::size() const{
	int ret = 0;
	for(int i = 0; i< matsNum; i++){
		ret += mats[i]->size();
	}
	return ret;
}
template <typename TYPE, bool CUDA >
MatGroup<TYPE, CUDA>::MatGroup(){
	matsNum = 0;
	mats = NULL;
	selfSpace = false;
	fixMat = false;
}
template <typename TYPE, bool CUDA >
MatGroup<TYPE, CUDA>::~MatGroup(){	
	memFree();
}
template <typename TYPE, bool CUDA >
void MatGroup<TYPE, CUDA>::memFree(){	
	if(mats != NULL){
		if(selfSpace){
			for(int i = 0; i< matsNum; i++){
				delete mats[i];
			}
		}
		delete [] mats;
	}
}
template <typename TYPE, bool CUDA >
void MatGroup<TYPE, CUDA>::copy(const MatGroup<TYPE, CUDA> & m){
	matsNum = m.matsNum;
	mats = new MatriX<TYPE, CUDA> *[matsNum];
	for(int i = 0; i< matsNum; i++){
		mats[i] = new MatriX<TYPE, CUDA>(*m.mats[i]);
	}
}

template <typename TYPE, bool CUDA >
MatGroup<TYPE, CUDA>::MatGroup<TYPE, CUDA>(const MatGroup<TYPE, !CUDA> & m){	
	selfSpace = true;
	matsNum = m.matsNum;
	fixMat = false;
	mats = new MatriX<TYPE, CUDA> *[matsNum];
	for(int i = 0; i< matsNum; i++){		
		mats[i] = new MatriX<TYPE, CUDA>(*m.mats[i]);		
	}

}
template <typename TYPE, bool CUDA >
MatGroup<TYPE, CUDA>::MatGroup<TYPE, CUDA>(const MatGroup<TYPE, CUDA> & m){	
	fixMat = false;
	selfSpace = true;
	copy(m);
}
template <typename TYPE, bool CUDA >
MatGroup<TYPE, CUDA>::MatGroup<TYPE, CUDA>(MatriX<TYPE, CUDA> * m, int num){
	matsNum = num;
	fixMat = false;
	selfSpace = false;
	mats = new MatriX<TYPE, CUDA> *[matsNum];
	for(int i = 0; i< matsNum; i++){		
		mats[i] = m + i;		
	}
};
template <typename TYPE, bool CUDA >
MatGroup<TYPE, CUDA>::MatGroup<TYPE, CUDA>(MatriX<TYPE, !CUDA> * m, int num){
	matsNum = mum;
	fixMat = false;
	selfSpace = true;
	mats = new MatriX<TYPE, CUDA> *[matsNum];
	for(int i = 0; i< matsNum; i++){		
		mats[i] = new MatriX<TYPE, CUDA>(m[i]);		
	}
};

template <typename TYPE, bool CUDA >
MatGroup<TYPE, CUDA> & MatGroup<TYPE, CUDA>::operator <<(MatriX<TYPE, CUDA> & mat){
	if(fixMat){
		Assert("无法在固定矩阵组中加入新矩阵");
	}
	MatriX<TYPE, CUDA> ** temp = new MatriX<TYPE, CUDA> *[matsNum + 1];
	if(matsNum > 0){
		memcpy(temp, mats, sizeof(MatriX<TYPE, CUDA> *) * matsNum);
	}
	temp[matsNum] = &mat;
	matsNum ++;
	if(mats != NULL){
		delete  [] mats;
	}
	mats = temp;
	return *this;
}
template <typename TYPE, bool CUDA >
MatGroup<TYPE, CUDA> & MatGroup<TYPE, CUDA>::operator += (const MatGroup<TYPE, CUDA> & m){
	if(fixMat && m.fixMat){
		fixMemMat += m.fixMemMat;
	}else{
		for(int i = 0; i< matsNum; i++){
			*mats[i] += *m.mats[i];
		}
	}
	return *this;
}
template <typename TYPE, bool CUDA >
MatGroup<TYPE, CUDA> & MatGroup<TYPE, CUDA>::operator -= (const MatGroup<TYPE, CUDA> & m){
	if(fixMat && m.fixMat){
		fixMemMat -= m.fixMemMat;
	}else{
		for(int i = 0; i< matsNum; i++){
			*mats[i] -= *m.mats[i];
		}
	}
	return *this;
}
template <typename TYPE, bool CUDA >
MatGroup<TYPE, CUDA>  MatGroup<TYPE, CUDA>::operator + (const MatGroup<TYPE, CUDA> & m) const{
	MatGroup<TYPE, CUDA> ret(*this);
	if(fixMat && m.fixMat){
		ret.fixMemMat +=  m.fixMemMat;
	}else{
		for(int i = 0; i< matsNum; i++){
			*ret.mats[i] += *m.mats[i];
		}
	}
	return ret;
}
template <typename TYPE, bool CUDA >
MatGroup<TYPE, CUDA>  MatGroup<TYPE, CUDA>::operator - (const MatGroup<TYPE, CUDA> & m) const{
	MatGroup<TYPE, CUDA> ret(*this);
	if(fixMat && m.fixMat){
		ret.fixMemMat -=  m.fixMemMat;	
	}else{
		for(int i = 0; i< matsNum; i++){
			*ret.mats[i] -= *m.mats[i];
		}
	}
	return ret;
}
template <typename TYPE, bool CUDA >
MatGroup<TYPE, CUDA> & MatGroup<TYPE, CUDA>::operator = (const MatGroup<TYPE, CUDA> & m){
	if(matsNum == 0){
		selfSpace = true;
		copy(m);
		return *this;
	}

	if(fixMat && m.fixMat ){
		if(fixMemMat.size() != m.fixMemMat.size()  || matsNum != m.matsNum){
			Assert("\n【错误】MatGroup大小不一致，无法赋值");
		}else{
			cuWrap::memD2D(fixMemMat.dataPrt(), m.fixMemMat.dataPrt(), sizeof(TYPE) * fixMemMat.size());
		}
	}else{
		if(matsNum != m.matsNum){
			Assert("\n【错误】MatGroup元素个数不一致，无法赋值");
		}
		for(int i = 0; i< matsNum; i++){	
			*mats[i] = m[i];
		}
	}
	return *this;
}
template <typename TYPE, bool CUDA >
MatGroup<TYPE, CUDA> & MatGroup<TYPE, CUDA>::operator /= ( TYPE scl){
	if(fixMat){
		cuWrap::scale(fixMemMat.dataPrt(),(TYPE)1.0f/scl, fixMemMat.size());	
	}else{
		for(int i = 0; i< matsNum; i++){
			*mats[i] /= scl;
		}
	}
	return *this;
}
template <typename TYPE, bool CUDA >
MatGroup<TYPE, CUDA> & MatGroup<TYPE, CUDA>::operator *= ( TYPE scl){
	if(fixMat){
		cuWrap::scale(fixMemMat.dataPrt(), scl, fixMemMat.size());	
	}else{
		for(int i = 0; i< matsNum; i++){
			*mats[i] *= scl;
		}
	}
	return *this;
}
template <typename TYPE, bool CUDA >
MatGroup<TYPE, CUDA> & MatGroup<TYPE, CUDA>::operator = ( TYPE val){
	if(fixMat){
		cuWrap::fill(fixMemMat.dataPrt(), val, fixMemMat.size());	
	}else{
		for(int i = 0; i< matsNum; i++){
			*mats[i] = val;
		}
	}
	return *this;
}
template <typename TYPE, bool CUDA >
MatGroup<TYPE, CUDA>  MatGroup<TYPE, CUDA>::operator * ( TYPE scl) const{
	MatGroup<TYPE, CUDA> ret(*this);
	ret *= scl;
	return ret;
}
template <typename TYPE, bool CUDA >
MatGroup<TYPE, CUDA>  MatGroup<TYPE, CUDA>::operator / ( TYPE scl) const{
	MatGroup<TYPE, CUDA> ret(*this);
	ret /= scl;
	return ret;
}
template <typename TYPE, bool CUDA >
MatriX<TYPE, CUDA> & MatGroup<TYPE, CUDA>::operator [] (int i) const{
	return *mats[i];
}
template <typename TYPE, bool CUDA >
TYPE MatGroup<TYPE, CUDA>::squaredNorm() const{
	double ret = 0;
	if(fixMat){
		ret = fixMemMat.squaredNorm();		
	}else{
		for(int i = 0; i< matsNum; i++){
			ret += mats[i]->squaredNorm();
		}
	}
	return ret;
}
template <typename TYPE, bool CUDA >
TYPE MatGroup<TYPE, CUDA>::norm() const{
	return sqrt(squaredNorm());
}
template <typename TYPE, bool CUDA >
TYPE MatGroup<TYPE, CUDA>::dot(const MatGroup<TYPE, CUDA> & m) const{
	double ret=0;
	if(fixMat && m.fixMat){
		ret = fixMemMat.dot(m.fixMemMat);
	}else{
		for(int i = 0; i< matsNum; i++){
			ret += mats[i]->dot(m[i]);
		}
	}
	return ret;
}
template <typename TYPE, bool CUDA >
TYPE MatGroup<TYPE, CUDA>::sum() const{
	double ret=0;
	if(fixMat){
		ret = fixMemMat.allSum();
	}else{
		for(int i = 0; i< matsNum; i++){
			ret += mats[i]->allSum();
		}
	}
	return ret;
}
template <typename TYPE, bool CUDA >
TYPE MatGroup<TYPE, CUDA>::correl(const MatGroup<TYPE, CUDA> & m) const{
	double ysum = 0; 
	double tsum = 0;
	double _dot = 0;
	double ysn = 0;
	double tsn = 0;
	int n = 0;
	for(int i = 0 ; i < matsNum; i++){
		ysum += mats[i]->allSum();
		tsum += m.mats[i]->allSum();
		_dot += mats[i]->dot(*m.mats[i]);
		ysn += mats[i]->squaredNorm();
		tsn += m.mats[i]->squaredNorm();
		n += mats[i]->size();
	}
	return  (n * _dot - ysum * tsum)/sqrt(( n * ysn - ysum * ysum) *( n * tsn - tsum * tsum));	
}
template <typename TYPE, bool CUDA >
TYPE MatGroup<TYPE, CUDA>::MSE() const{
	double avg = 0; 
	int n = 0;
	for(int i = 0 ; i < matsNum; i++){
		avg += mats[i]->allSum();
		n += mats[i]->size();
	}
	avg /= n;
	double mse = 0; 
	for(int i = 0 ; i < matsNum; i++){
		mse += (*mats[i] - (TYPE)avg).squaredNorm();		
	}
	return  mse/n;	
}
template <typename TYPE, bool CUDA >
void MatGroup<TYPE, CUDA>::show() const{
	cout<<"\nnum:"<<matsNum<<"\n";
	for(int i = 0; i< matsNum; i++){
		cout<<"["<<i<<"]"<<mats[i]->str();
	}
}
template <typename TYPE,bool CUDA >
MatGroup<TYPE, CUDA>  operator * (const TYPE  scl, const MatGroup<TYPE, CUDA> & m){
	MatGroup<TYPE, CUDA> ret(m);
	ret *= scl;
	return ret;
}
template <typename TYPE,bool CUDA >
MatGroup<TYPE, CUDA>  operator - (const MatGroup<TYPE, CUDA> & m){
	MatGroup<TYPE, CUDA> ret(m);
	ret *= -1;
	return ret;
}
template <typename TYPE, bool CUDA >
void MatGroup<TYPE, CUDA>::save(ofstream & fl) const{
	fl.write((char *)&matsNum, sizeof(int));
	fl.write((char *)&fixMat, sizeof(bool));
	for(int i = 0; i< matsNum; i++){
		mats[i]->save(fl);
	}
}
template <typename TYPE, bool CUDA >
void MatGroup<TYPE, CUDA>::read(ifstream & fl){
	fl.read((char *)&matsNum, sizeof(int));
	bool tmp;
	fl.read((char *)&tmp, sizeof(bool));	
	for(int i = 0; i< matsNum; i++){
		mats[i]->read(fl);
	}
	if(tmp){
		setFix();
	}
}
template <typename TYPE, bool CUDA >
void MatGroup<TYPE, CUDA>::setFix(){
	if(CUDA && !fixMat){
		fixMemMat = MatriX<TYPE, CUDA>::Zero(size(),1);
		int pos = 0;
		for(int i = 0; i< matsNum; i++){
			pos += mats[i]->size();
		}
		fixMat = true;
	}
}
template <typename TYPE, bool CUDA >
void MatGroup<TYPE, CUDA>::fixFree(){
	if(fixMat){
		for(int i = 0; i< matsNum; i++){
				
		}
		fixMemMat = MatriX<TYPE, CUDA>::Zero(0,0);
		fixMat = false;
	}
}
template <typename TYPE, bool CUDA >
MatGroup<TYPE, CUDA> & MatGroup<TYPE, CUDA>::clear(){
	memFree();
	matsNum = 0;
	mats = NULL;
	selfSpace = false;
	fixMat = false;
	return *this;
}
template <typename TYPE, bool CUDA >
int MatGroup<TYPE, CUDA>::num() const{
	return matsNum;
}