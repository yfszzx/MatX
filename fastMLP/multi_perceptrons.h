class multi_perceptrons{
public:
	int input_dimen;
	int output_dimen;
	int layers_num;
	int weight_len;
	float *weight;	
	float *deriv;
	float result;
	float real_result;
	bool train_mod;
	char loss_mod;	
	bool reguler_mod;
	bool save_history;
	perceptrons *nerv_bottom,*nerv_top,**nervs;
	int data_num;
	string path;
	string proj_path;
	string file_name;
	multi_perceptrons(string name);
		void init_nervs();
	void struct_init();
	void struct_settted(bool save=true);
	void weight_rand();
	void set_data_num(int dt_num);
	float get_result(float *input,float *target);
	float *layer_out(int layer,float *&s_input,int num,bool in_cuda_pos=true);
	float run(float *s_input,float *out,float *s_target,int num,float dropout=1.0f,bool in_cuda_pos=true,bool out_cuda_pos=true);
	void top_pre_deriv();
	void cacul_deriv(float *input,float *target);
	void cacul_nerv(float *input,float *target);
	void struct_show();
	bool struct_read();
	void struct_save(int idx=-1);
	void reguler_set();
	void loss_mod_set();
	void reguler_show();
	void struct_set(int i_dimen=0,int o_dimen=0);
	void struct_simple_set(int in_dimen,int out_dimen,int nodes,char nodes_mod,char out_mod,char loss='2',float  decay=0,char decay_mod='2');
	int select_layer(string illu="");
	void struct_edit();

	~multi_perceptrons();
	int memery(bool show=true);
	void reset();	
	private:
	float *output_tmp;
	float *tmp_array;
	void free_mem();
	int o_len;
	int blocks;
	int ci;
};