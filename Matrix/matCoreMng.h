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
	if(!CUDA){
		init(rowsNum, colsNum);		
		load(prt, false);
	}else{
		if(fix()){
			Assert("已经是fixMem矩阵，不能重新设置");
		}		
		scale = 1;
		transFlag = false;
		free();
		mem = new matMem<TYPE, CUDA>(prt);
	}
}
template <typename TYPE, bool CUDA>
void matCore<TYPE, CUDA>::quoteLoad(const matMem<TYPE, CUDA> * m){
	free();
	mem = m->mem;
	mem->count ++;
}
template <typename TYPE, bool CUDA>
void  matCore<TYPE, CUDA>::copy(const MatriX<TYPE, CUDA> &m){
	if(this == &m){
		return ;
	}
	if(fix()){
		if(rowsNum != m.rowsNum || colsNum != m.colsNum){
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
void  matCore<TYPE, CUDA>::copy(const MatriX<TYPE, !CUDA> &m){
	if(fix()){
		if(rowsNum != m.rowsNum || cols != m.colsNum){
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
			int t = rowsNum;
			rowsNum = colsNum;
			colsNum = t;
		}
		transFlag = false;
		scale = 1;
	}	
}
template <typename TYPE, bool CUDA>
void matCore<TYPE, CUDA>::copyRealise(bool sclRealise, bool trnRealise){
	float tScl = scale;
	if(!unique()){
		bool tTrans = transFlag;
		matMem *tMem = *mem;
		init(rows, cols);
		transFlag = tTrans;
		load(tMem);
	}
	if(sclRealise){
		scaleRealise();
	}else{
		scale = tScl;
	}
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
