//构造函数
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA>::MatriX(int _rows,int _cols){
	init(_rows, _cols);
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA>::MatriX(const MatriX<TYPE,CUDA> &m){
	copy(m);
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA>::MatriX(const MatriX<TYPE,!CUDA> &m){
	copy(m);
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA>::MatriX(const MatriX<TYPE, CUDA> &m, operateType OPT){
		copy(m);
		switch (OPT){
		case STR:
			memRealise();
			break;
		case TRN:
			copyRealise(false, true);
			break;
		case SCL:
			copyRealise(true, false);
			break;
		case ALL:
			copyRealise(true, true);
			break;
		}
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA>::MatriX(const MatriX<TYPE, CUDA> &m, TYPE * prt){
	copy(m, prt);	
}
//赋值函数
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> & MatriX< TYPE,CUDA>::operator = (const MatriX< TYPE, CUDA> &m){
	copy(m);
	return *this;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> & MatriX< TYPE,CUDA>::operator = (const MatriX< TYPE, !CUDA> &m){
	copy(m);
	return *this;
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> & MatriX< TYPE,CUDA>::operator = (const TYPE val){
	memRealise();
	if(CUDA){
		cuWrap::fill(dataPrt(), val, size());
	}else{
		std::fill(dataPrt(), dataPrt() + size(), val);
	}
	return *this;
}

template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> MatriX< TYPE,CUDA>::Random(int _rows, int  _cols){
	MatriX< TYPE,CUDA> ret(_rows, _cols);
	return ret.selfRandom();
}; 
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> MatriX< TYPE,CUDA>::Constant(int _rows, int  _cols, TYPE val){
	MatriX<TYPE, CUDA> ret(_rows, _cols);
	ret = val;
	return ret;
}; 
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> MatriX< TYPE,CUDA>::Ones(int _rows, int  _cols){
	MatriX< TYPE,CUDA> ret(_rows, _cols);
	ret = 1;
	return ret;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> MatriX< TYPE,CUDA>::Zero(int _rows, int  _cols){
	MatriX< TYPE,CUDA> ret(_rows, _cols);
	ret = 0;
	return ret;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> MatriX< TYPE,CUDA>::eye(int _rows, int  _cols){
	if(_cols == 0){
		_cols = _rows;
	}
	if(!CUDA){
			MatriX< TYPE,CUDA> ret(_rows, _cols);
			ret.eMat() = eigenMat::Identity(_rows, _cols);
			return ret;
	}else{
		eigenMat tmp = eigenMat::Identity(_rows, _cols);
		MatriX< TYPE,CUDA> ret(_rows, _cols);
		ret.loadMat(tmp.data(),false);
		return ret;
	}
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> MatriX< TYPE,CUDA>::Identity(int _rows, int  _cols){
	return eye(_rows, _cols);
}
template <typename TYPE, bool CUDA>
MatriX< TYPE, CUDA> &MatriX< TYPE,CUDA>::selfRandom(){
	if(CUDA){
		if(randDebug){
			cout<<"\nCUDA的随机数生成处于调试状态，将matrixGlobal中的randDebug设为false恢复正常";
			eigenMat tmp = eigenMat::Random(rows(), cols());	
			cuWrap::memH2D(dataPrt(), tmp.data(), sizeof(TYPE) * size());
		}else{
		cuWrap::random(dataPrt(), size());
		*this *= 2;
		*this += -1;
		}
	}else{
		eMat() = eigenMat::Random(rows(), cols());	
	}

	return *this;
}
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> & MatriX< TYPE,CUDA>::importData(TYPE * src, bool cuda){
	loadMat(src, cuda);
	return *this;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> & MatriX<TYPE,CUDA>::assignment(int row, int col, TYPE val){
	assignment(elementPos(row, col), val);
	return *this;
};
template <typename TYPE, bool CUDA>
MatriX<TYPE,CUDA> & MatriX<TYPE,CUDA>::assignment(int idx, TYPE val){
	idx = realPos(idx);
	val/=scale;
	if(CUDA){
		cuWrap::memH2D(dataPrt() + idx, &val, sizeof(TYPE));
	}else{
		dataPrt()[idx] = val;
	}
	return *this;
};


