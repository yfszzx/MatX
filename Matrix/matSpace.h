template <typename TYPE, bool CUDA>
class  matSpace{
private:
	TYPE * body;
	Matrix<TYPE, -1, -1> *eigen_mat; 
	int size;
	int count;
	bool fixMem;
public:

	inline void copy_data(const  matSpace<TYPE, CUDA> & m);
	inline void copy_data(const ( matSpace<TYPE, !CUDA> & m);
	inline void copy_data(TYPE * src, bool cuda);
	void scale(TYPE scl);
	void transpose(int rows, int cols);
	inline bool destroy();
	inline void add();
	void setFixMem();
	inline bool isFixMem();
	inline bool isUnique();
	inline Matrix<TYPE, -1, -1> & eigenMat();
	inline TYPE * data();
	 matSpace(int rows, int cols);
	 matSpace();
	~ matSpace();
	
};

template <typename TYPE, bool CUDA>
void  matSpace<TYPE, CUDA>::copy_data(const  matSpace<TYPE, CUDA> & m){
	copy_data(m.data(), CUDA);	
}
template <typename TYPE, bool CUDA>
void  matSpace<TYPE, CUDA>::copy_data(const  matSpace<TYPE, !CUDA> & m){
	copy_data(m->data(), !CUDA);		
}
template <typename TYPE, bool CUDA>
void  matSpace<TYPE, CUDA>::copy_data(TYPE * src, bool cuda){
	if(count == 0 || data() == NULL){
		cout<<"\n【错误】空间为空\n";
		return;
	}
	if(count > 1){
		cout<<"\n【错误】空间存在副本(count:"<<count<<")，不能为空间赋值\n";
		return;
	}
	if(cuda){
		if(CUDA){
			cuWrap::memD2D(data(), src, sizeof(TYPE) * size);
		}else{
			cuWrap::memD2H(data(), src, sizeof(TYPE) * size);
		}
	}else{
		if(CUDA){
			cuWrap::memH2D(data(), src, sizeof(TYPE) * size);
		}else{
			memcpy(data(), src, sizeof(TYPE) * size);
		}
	}

}
template <typename TYPE, bool CUDA>
 matSpace<TYPE, CUDA>:: matSpace(int rows, int cols){
	size = rows * cols;
	count = 1;
	fixMem = false;
	if(size == 0){
		body = NULL;
		eigen_mat = NULL;
	}else{
		if (CUDA){
			cuWrap::malloc((void **)&body, sizeof(TYPE) * size);
		}else{
			eigen_mat = new Matrix<TYPE, -1, -1>(rows, cols);
		}
	}	
}
template <typename TYPE, bool CUDA>
 matSpace<TYPE, CUDA>:: matSpace(){
	size = 0;
	count = 0;
	fixMem = false;
	body = NULL;
	eigen_mat = NULL;
	
}
template <typename TYPE, bool CUDA>
void  matSpace<TYPE, CUDA>::setFixMem(TYPE * prt){
	if(CUDA){
		if(fixMem && data() != prt){
			cout<<"【错误】已经是固定空间不能重新设置\n";
			getchar();
		}else{
			if(count > 1){
				cout<<"【错误】存在副本，不能设为固定空间\n";
				getchar();
			}		
			if(count == 1){
				cuWrap:memD2D(prt, body, sizeof(TYPE) * size);
				cuWrap::free(body);
			}
			body = prt;
			count = 1;
			fixMem = true;
		}
	}
}
template <typename TYPE, bool CUDA>
bool  matSpace<TYPE, CUDA>::isFixMem(){
	return fixMem;
}
template <typename TYPE,  CUDA>::bool isUnique(){
	return count == 1;
}
template <typename TYPE, bool CUDA>
 matSpace<TYPE, CUDA>::~ matSpace(){
	if(count>0){
		cout<<"\n【错误】空间有副本存在，不能销毁\n";
		getchar();
	}
}
template <typename TYPE, bool CUDA>
bool  matSpace<TYPE, CUDA>::destroy(){		
	count--;
	if(data() == NULL){
		return count == 0;
	}
	if(count == 0){
		if (CUDA){
			cuWrap::free(body);		
		}else{
			delete eigen_mat;
		}
		body = NULL;
		eigen_mat = NULL;
	}
	return count == 0;
}
template <typename TYPE, bool CUDA>
bool  matSpace<TYPE, CUDA>::add(){		
	count ++ ;
}
template <typename TYPE, bool CUDA>
Matrix<TYPE, -1, -1> &  matSpace<TYPE, CUDA>::eigenMat(){		
	return *eigen_mat;
}
template <typename TYPE, bool CUDA>
TYPE *  matSpace<TYPE, CUDA>::data(){		
	if(CUDA){
		return body;
	}else{
		return eigen_mat->data();
	}
}
template <typename TYPE, bool CUDA>
void  matSpace<TYPE, CUDA>::scale(TYPE scl){	
	if(scl == 1)return;
		if(CUDA){
			cuWrap::scale(body, scl, size);
		}else{
			*eigen_mat *=  t_scl;
		}
}
template <typename TYPE, bool CUDA>
void  matSpace<TYPE, CUDA>::transpose(int rows, int cols){
	if(CUDA){
		cuWrap::transpose(rows, cols, body,  size);
	}else{
		*eigen_mat = eigen_mat->transpose();
	}
}