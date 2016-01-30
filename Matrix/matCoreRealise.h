template <typename TYPE, bool CUDA>
void matCore<TYPE, CUDA>::free(){
	if(mem->count> 1){
		mem->count --;
	}else{
		delete mem;
	}
}
template <typename TYPE, bool CUDA>
matCore<TYPE, CUDA>::matCore(){
	mem = new matMem<TYPE, CUDA>();
	sizeNum = 0;
	rowsNum = 0;
	colsNum = 0;
}

template <typename TYPE, bool CUDA>
matCore<TYPE, CUDA>::~matCore(){
	free();
}
template <typename TYPE, bool CUDA>
void matCore<TYPE, CUDA>::init(int _rows, int _cols){
	if(fix()){
		Assert("试图改变fixMem的矩阵");
	}
	rowsNum = _rows;
	colsNum = (_rows == 0)?0:_cols;
	scale = 1;
	transFlag = false;
	if(!unique() || sizeNum != _rows *  _cols){	
		//空间大小不同或者并非原始空间时，就需要开辟新的原始空间
		sizeNum = rowsNum * colsNum;
		free();
		mem = new matMem<TYPE, CUDA>(rowsNum, colsNum);
	}
}
template <typename TYPE, bool CUDA>
void matCore<TYPE, CUDA>::setFix(TYPE * prt){
	if(CUDA){
		if(fix()){
			Assert("已经是fixMem矩阵，不能重新设置");
		}		
		copyRealise(true, true);
		cuWrap::memD2D(prt, dataPrt(), sizeof(TYPE) * size());
		free();
		mem = new matMem<TYPE, CUDA>(prt);
	}
}
template <typename TYPE, bool CUDA>
void matCore<TYPE, CUDA>::fixFree(){
	if(fix()){
	matMem * tmp = mem;	
	mem  = new matMem<TYPE, CUDA>(rowsNum, colsNum);
	cuWrap::memD2D(mem->mem, tmp->mem,sizeof(TYPE) * size());
	delete tmp;
	}
}
template <typename TYPE, bool CUDA>
void matCore<TYPE, CUDA>::quoteLoad(const matMem<TYPE, CUDA> * m){
	free();
	mem =  const_cast<matMem<TYPE, CUDA> *>(m);
	mem->count ++;
}
template <typename TYPE, bool CUDA>
void  matCore<TYPE, CUDA>::copy(const MatriX<TYPE, CUDA> &m){
	if((void *)this == (void *)&m){
		return ;
	}
	if(fix()){
		if(rowsNum != m.realRows() || colsNum != m.realCols()){
			Assert("fixMem的目标矩阵与输入矩阵维度不一致，无法复制");
		}
		load(m.mem);		
	}else{
		if(m.fix()){
			init(m.rowsNum, m.colsNum);		
			load(m.mem);
		}else{
			rowsNum = m.rowsNum;
			colsNum = m.colsNum;
			sizeNum = m.sizeNum;	
			quoteLoad(m.mem);
		}		
	}
	scale = m.scale;
	transFlag = m.transFlag;	
}
template <typename TYPE, bool CUDA>
void  matCore<TYPE, CUDA>::copy(const MatriX<TYPE, CUDA> &m, TYPE * prt){
			rowsNum = m.rowsNum;
			colsNum = m.colsNum;
			sizeNum = m.sizeNum;
			scale =m.scale;
			transFlag = m.transFlag;
			mem = new matMem<TYPE, CUDA>(prt);
}
template <typename TYPE, bool CUDA>
void  matCore<TYPE, CUDA>::tmpcopy(const MatriX<TYPE, CUDA> &m){
	if((void *)this == (void *)&m){
		return ;
	}
	rowsNum = m.rowsNum;
	colsNum = m.colsNum;
	sizeNum = m.sizeNum;	
	quoteLoad(m.mem);
	scale = m.scale;
	transFlag = m.transFlag;	
}
template <typename TYPE, bool CUDA>
void  matCore<TYPE, CUDA>::copy(const MatriX<TYPE, !CUDA> &m){
	if(fix()){
		if(rowsNum != m.realRows() || colsNum != m.realCols()){
			Assert("fixMem的目标矩阵与输入矩阵维度不一致，无法复制");
		}		
	}else{
		init(m.rowsNum, m.colsNum);	
	}
	load(m.mem);
	scale =m.scale;
	transFlag = m.transFlag;

}
template <typename TYPE, bool CUDA>
void matCore<TYPE, CUDA>::memRealise(){
	if(!unique()){	
		init(rows(), cols());
	}else{
		if(transFlag){
			swap(rowsNum, colsNum);
		}
		transFlag = false;
		scale = 1;
	}	
}
template <typename TYPE, bool CUDA>
void matCore<TYPE, CUDA>::copyRealise(bool sclRealise, bool trnRealise){
	
	if(!unique()){
		float tScl = scale;
		bool tTrans = transFlag;
		matMem<TYPE, CUDA> *tMem = mem;
		init(rowsNum, colsNum);
		transFlag = tTrans;
		load(tMem);
		scale = tScl;
	}	
	if(sclRealise){
		scaleRealise();
	};
	if(trnRealise){
		transposeRealise();
	}
}

template <typename TYPE, bool CUDA>
void matCore<TYPE, CUDA>::load(const TYPE * src, bool cuda = false){
	if(cuda){
		if(CUDA){
			cuWrap::memD2D(dataPrt(), src, sizeof(TYPE) * size());
		}else{
			cuWrap::memD2H(dataPrt(), src, sizeof(TYPE) * size());
		}
	}else{
		if(CUDA){
			cuWrap::memH2D(dataPrt(), src, sizeof(TYPE) * size());
		}else{
			memcpy(dataPrt(), src, sizeof(TYPE) * size());
		}
	}
}

template <typename TYPE, bool CUDA>
void matCore<TYPE, CUDA>::load(const matMem<TYPE, CUDA> *m){
	load(m->mem, CUDA);
}
template <typename TYPE, bool CUDA>
void matCore<TYPE, CUDA>::load(const matMem<TYPE, !CUDA> *m){
	load(m->mem, !CUDA);
}

template <typename TYPE, bool CUDA>
int matCore<TYPE, CUDA>::cols() const{
	if(transFlag){
		return rowsNum;
	}else{
		return colsNum;
	}	
}
template <typename TYPE, bool CUDA>
int matCore<TYPE, CUDA>::rows() const{
	if(transFlag){
		return colsNum;
	}else{
		return rowsNum;
	}	
}
template <typename TYPE, bool CUDA>
int matCore<TYPE, CUDA>::realCols() const{
	return colsNum;
}
template <typename TYPE, bool CUDA>
int matCore<TYPE, CUDA>::realRows() const{
	return rowsNum;
}
template <typename TYPE, bool CUDA>
int matCore<TYPE, CUDA>::size() const{
	return sizeNum;	
}
template <typename TYPE, bool CUDA>
TYPE * matCore<TYPE, CUDA>::dataPrt(){
	return mem->mem;	
}
template <typename TYPE, bool CUDA>
TYPE * matCore<TYPE, CUDA>::dataPrt() const{
	return mem->mem;	
}
template <typename TYPE, bool CUDA>
eigenMat matCore<TYPE, CUDA>::eMat() const{
	return *mem->eigenMem;	
}
template <typename TYPE, bool CUDA>
eigenMat & matCore<TYPE, CUDA>::eMat() {
	return *mem->eigenMem;	
}
template <typename TYPE, bool CUDA>
bool matCore<TYPE, CUDA>::unique() const{
	return (mem->count  == 1);
}
template <typename TYPE, bool CUDA>
bool matCore<TYPE, CUDA>::fix() const{
	return mem->fixMem;
}
template <typename TYPE, bool CUDA>
void  matCore<TYPE, CUDA>::transposeRealise(){
	if(!transFlag){	
		return;
	}
	if(size() == 1){
		transFlag = false;
		return;
	}
	if(!unique()){
		Assert("无法改变存在副本的矩阵");
	}
	if(CUDA){
		cuWrap::transpose(rowsNum, colsNum, dataPrt(),  sizeNum);
	}else{
		eigenMat tmp = eMat().transpose();
		eMat() = tmp;
	}
	transFlag = false;
}
template <typename TYPE, bool CUDA>
void  matCore<TYPE, CUDA>::scaleRealise(){	
	if(scale == 1){
		return;
	}
	if(!unique()){
		Assert("无法改变存在副本的矩阵");
	}
	if(CUDA){
		cuWrap::scale(dataPrt(), scale, size());
	}else{
		eMat() *=  scale;
	}
	scale = 1;
}
template <typename TYPE, bool CUDA>
bool  matCore<TYPE, CUDA>::trans() const{
	return transFlag;
}
template <typename TYPE, bool CUDA>
void  matCore<TYPE, CUDA>::setTrans(){
	transFlag = !transFlag;
	if(size() == 1){
		transFlag = false;
	}
}
template <typename TYPE, bool CUDA>
void  matCore<TYPE, CUDA>::setTrans(bool flag){
	transFlag = flag;
	if(size() == 1){
		transFlag = false;
	}
}
template <typename TYPE, bool CUDA>
void  matCore<TYPE, CUDA>::loadMat(const TYPE * src, bool cuda){
	memRealise();
	load(src, cuda);
}
template <typename TYPE, bool CUDA>
string matCore<TYPE, CUDA>::str() const{
	stringstream ret;
	ret<<"\ncols:"<<cols()<<" rows:"<<rows()<<" size:"<<size()<<" memery:"<<size() * sizeof(TYPE)<<" prtCount:"<<mem->count<<"\n";
	ret<<"cuda:"<<CUDA<<" dataType:"<<typeid(TYPE).name()<<" trans:"<<transFlag<<" scale:"<<scale<<" fix:"<<fix()<<"\n";
	return ret.str();	
};





