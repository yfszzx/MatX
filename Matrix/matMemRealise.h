template <typename TYPE, bool CUDA>
matMem<TYPE, CUDA>:: matMem(){
	count = 1;
	fixMem = false;
	mem = NULL;
	eigenMem = NULL;	
}
template <typename TYPE, bool CUDA>
matMem<TYPE, CUDA>::matMem(int rows, int cols){
	count = 1;
	fixMem = false;
	int size = rows * cols;
	if(size == 0){
		mem = NULL;
		eigenMem = NULL;
	}else{
		if (CUDA){
			cuWrap::malloc((void **)&mem, sizeof(TYPE) * size);
			eigenMem = NULL;
		}else{
			eigenMem = new eigenMat(rows, cols);
			mem = eigenMem->data();
		}
	}	
}
template <typename TYPE, bool CUDA>
matMem<TYPE, CUDA>:: matMem(TYPE * prt){
	if(!CUDA){
		Assert("cpu矩阵无法设置为fixMem模式");
	}
	count = 1;
	fixMem = true;
	mem = prt;	
}
template <typename TYPE, bool CUDA>
matMem<TYPE, CUDA>:: ~matMem(){
	if(count>1){
		cout<<"\ncount:"<<count;
		Assert("matMem有副本存在，不能销毁\n");
	}
	if(CUDA){
		if(mem != NULL && !fixMem){
			cuWrap::free(mem);
		}
	}else{
		if(eigenMem != NULL){
			delete eigenMem;
		}
	}	
}