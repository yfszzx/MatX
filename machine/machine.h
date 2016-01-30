#include <conio.h>   //¼àÌý¼üÅÌ
#include <time.h>
#include "machineGlobal.h"
#include "dataSetBase.h"
#include "searchTool.h"
#include "machineBase.h"
#include "ANNBase.h"
#include "MLP.h"
#include "seriesDataBase.h"
#include "RNN.h"
#include "ELM.h"
#include "ESN.h"

#include "dataSetBaseRealise.h"
#include "searchToolRealise.h"
#include "machineBaseRealise.h"
#include "ANNBaseRealise.h"
#include "MLPRealise.h"
#include "seriesDataBaseRealise.h"
#include "RNNRealise.h"
#include "ELMRealise.h"
#include "samples.h"
typedef MLP<double, true> ANNGD;
typedef MLP<double, false> ANNCD;
typedef MLP<float, true> ANNGF;
typedef MLP<float, false> ANNCF;

