Search.setIndex({docnames:["code","index"],envversion:53,filenames:["code.rst","index.rst"],objects:{"":{basic:[0,0,0,"-"],bodies:[0,0,0,"-"],db_handling:[0,0,0,"-"],generators:[0,0,0,"-"],heads:[0,0,0,"-"],model_ensemble:[0,0,0,"-"],model_manager:[0,0,0,"-"],pre_post:[0,0,0,"-"],pre_post_fns:[0,0,0,"-"],signal_proc:[0,0,0,"-"],testing:[0,0,0,"-"],training:[0,0,0,"-"],uncertainty:[0,0,0,"-"],utils:[0,0,0,"-"]},"bodies.ImageData":{dcnn:[0,3,1,""],dcnn_upsampling:[0,3,1,""],dense_net:[0,3,1,""],nin:[0,3,1,""],pyramid_pooling_net:[0,3,1,""],resnet:[0,3,1,""],resnet_upsampling:[0,3,1,""],segnet:[0,3,1,""]},"bodies.ProcessData":{conv_leg:[0,3,1,""],feature_reduction_mlp:[0,3,1,""],memoriless_engineeredstats:[0,3,1,""],memoriless_mlp:[0,3,1,""],pca:[0,3,1,""],randomized_feature_bagger:[0,3,1,""],reducer_regressor:[0,3,1,""],split_features:[0,3,1,""],take_feature:[0,3,1,""],temporal_reduction_conv:[0,3,1,""]},"generators.HistorySampler":{__init__:[0,4,1,""],get_indices:[0,4,1,""],get_valid_index_range:[0,4,1,""]},"generators.Sampler":{get_indices:[0,4,1,""]},"generators.SpatioTemporal":{__init__:[0,4,1,""]},"generators.TableOfScalars":{__init__:[0,4,1,""]},"heads.Classification":{global_aggr_mlp:[0,3,1,""]},"model_ensemble.ModelEnsemble":{__init__:[0,4,1,""],ensemble_statistics:[0,5,1,""],predict:[0,4,1,""]},"model_manager.ModelManager":{__init__:[0,4,1,""],compile_model:[0,4,1,""],from_recipe:[0,3,1,""],save_recipe:[0,4,1,""],summary:[0,4,1,""]},"pre_post.Processor":{__call__:[0,4,1,""],__init__:[0,4,1,""],add_function:[0,4,1,""],column_normalize:[0,4,1,""],column_standardize:[0,4,1,""],extract_slag_signal:[0,4,1,""],from_recipe:[0,3,1,""],mask:[0,4,1,""],mask_image:[0,4,1,""],mean_normalize:[0,4,1,""],range_normalize:[0,4,1,""],spatial_transform:[0,4,1,""],standardize:[0,4,1,""],take_axis:[0,4,1,""]},"pre_post_fns.MaskImage":{__call__:[0,4,1,""],__init__:[0,4,1,""]},"pre_post_fns.PersistentPrePostFunction":{__call__:[0,4,1,""],__init__:[0,4,1,""]},"pre_post_fns.SlagSignalExtractor":{__call__:[0,4,1,""],__init__:[0,4,1,""]},"pre_post_fns.SpatialTransformAugmentation":{__call__:[0,4,1,""],__init__:[0,4,1,""]},"testing.ProcessSynchedTimeWithSensitivity":{__init__:[0,4,1,""],evaluate:[0,4,1,""]},"testing.ProcessTestingProtocol":{__init__:[0,4,1,""],evaluate:[0,4,1,""]},"testing.TestingProtocol":{__init__:[0,4,1,""],evaluate:[0,4,1,""]},"training.ImageAugmentingProtocol":{__init__:[0,4,1,""],compile_and_train:[0,4,1,""],get_idg:[0,4,1,""]},"training.ImageClassificationXYAugmenting":{__init__:[0,4,1,""],compile_and_train:[0,4,1,""]},"training.ImageSegmentationXYAugmenting":{compile_and_train:[0,4,1,""]},"training.ProcessInterpolatedInMemory":{__init__:[0,4,1,""],compile_and_train:[0,4,1,""]},"training.ProcessSynchedTimeSampler":{__init__:[0,4,1,""],compile_and_train:[0,4,1,""]},"training.TrainingProtocol":{__init__:[0,4,1,""],compile_and_train:[0,4,1,""]},"uncertainty.EnsemblePrediction":{__init__:[0,4,1,""],estimate_ci:[0,4,1,""],resample:[0,4,1,""]},"uncertainty.NormalStatistics":{__init__:[0,4,1,""],estimate_ci:[0,4,1,""],resample:[0,4,1,""]},basic:{compression:[0,1,1,""],conv2d:[0,1,1,""],input_layer:[0,1,1,""]},bodies:{ImageData:[0,2,1,""],ProcessData:[0,2,1,""]},db_handling:{clean_metadata:[0,1,1,""],clean_signal:[0,1,1,""],drop_tables:[0,1,1,""],explore_signals:[0,1,1,""],filter_metadata:[0,1,1,""],get_process_area:[0,1,1,""],is_signal_name_valid:[0,1,1,""],list_tables:[0,1,1,""],query_at:[0,1,1,""],query_between:[0,1,1,""],query_datetime_bound:[0,1,1,""],query_datetime_range:[0,1,1,""],read_signal:[0,1,1,""],sql_between:[0,1,1,""],timestring_to_datetime:[0,1,1,""],values:[0,1,1,""],visualize_signal:[0,1,1,""],visualize_signals:[0,1,1,""]},generators:{HistorySampler:[0,2,1,""],Sampler:[0,2,1,""],SpatioTemporal:[0,2,1,""],TableOfScalars:[0,2,1,""],identity_reducer_function:[0,1,1,""],test_images_image:[0,1,1,""],test_images_text:[0,1,1,""],test_reducer_function:[0,1,1,""],test_sequence_text:[0,1,1,""]},heads:{Classification:[0,2,1,""]},model_ensemble:{ModelEnsemble:[0,2,1,""]},model_manager:{ModelManager:[0,2,1,""]},pre_post:{Processor:[0,2,1,""],estimate_image_standardization:[0,1,1,""],estimate_process_standardization:[0,1,1,""],function_from_recipe_line:[0,1,1,""]},pre_post_fns:{MaskImage:[0,2,1,""],PersistentPrePostFunction:[0,2,1,""],SlagSignalExtractor:[0,2,1,""],SpatialTransformAugmentation:[0,2,1,""],mask_image:[0,1,1,""],normalize_columns:[0,1,1,""],rescale:[0,1,1,""],standardize:[0,1,1,""],standardize_columns:[0,1,1,""],subtract_mean:[0,1,1,""],sum_values:[0,1,1,""],take_channel:[0,1,1,""]},signal_proc:{find_contiguous_time:[0,1,1,""],linked_rate_plot:[0,1,1,""],plot_data:[0,1,1,""],scattered_average:[0,1,1,""],scattered_average_rate:[0,1,1,""],scattered_moving_average:[0,1,1,""]},testing:{ProcessSynchedTimeWithSensitivity:[0,2,1,""],ProcessTestingProtocol:[0,2,1,""],TestingProtocol:[0,2,1,""],gradient_sensitivity_analysis:[0,1,1,""]},training:{ImageAugmentingProtocol:[0,2,1,""],ImageClassificationXYAugmenting:[0,2,1,""],ImageSegmentationXYAugmenting:[0,2,1,""],ProcessInterpolatedInMemory:[0,2,1,""],ProcessSynchedTimeSampler:[0,2,1,""],TrainingProtocol:[0,2,1,""]},uncertainty:{EnsemblePrediction:[0,2,1,""],NormalStatistics:[0,2,1,""]},utils:{check_dataset:[0,1,1,""],datestr_to_timestamp:[0,1,1,""],destandardize:[0,1,1,""],filename_to_timestamp:[0,1,1,""],get_frame_number:[0,1,1,""],imoverlay:[0,1,1,""],list_folder_contents:[0,1,1,""],load_image_data:[0,1,1,""],log1l:[0,1,1,""],one_hot_encode:[0,1,1,""],read_image:[0,1,1,""],read_labels:[0,1,1,""],read_recipe:[0,1,1,""],relative_path:[0,1,1,""],save_history:[0,1,1,""],show_segmentation_training_data:[0,1,1,""],show_thumbnail:[0,1,1,""],standardize:[0,1,1,""],timestamp_to_datetime:[0,1,1,""],write_image:[0,1,1,""]}},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","staticmethod","Python static method"],"4":["py","method","Python method"],"5":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:staticmethod","4":"py:method","5":"py:attribute"},terms:{"0_class_0":0,"1_class_1":0,"1x2":0,"2x2":0,"abstract":[],"case":0,"class":0,"default":0,"final":0,"float":0,"function":0,"import":0,"int":0,"long":0,"new":0,"return":0,"static":0,"switch":0,"true":0,Axes:0,C1s:0,C2s:0,For:0,Has:0,Not:0,One:0,The:0,Then:0,There:0,These:0,Use:0,Used:0,Useful:0,__call__:0,__init__:0,__next__:0,_abbr:0,_ax:0,_histori:0,_required_column:0,_subplot:0,abbrevi:0,about:0,abov:0,absolut:0,accept:0,accur:0,accuraci:0,act:0,activ:0,actual:0,adam:0,add:[0,1],add_funct:0,added:0,addit:0,adjac:0,affect:0,after:0,aggr_fn:0,aggreg:0,alia:0,all:0,allow:0,allow_multiple_class:0,along:0,alpha:0,alreadi:1,also:0,amount:0,analysi:0,angl:0,ani:0,anoth:0,apart:0,append:0,appli:0,appropri:0,approxim:0,arbitrari:0,architectur:0,area:0,arg:0,argument:0,around:0,arrai:[0,1],assess:0,assum:0,attach:0,attempt:0,aug_arg:0,augment:0,autoencod:0,aux_input:0,avail:0,averag:[0,1],avg:0,axes:0,axi:0,back:0,backend:0,background:0,backward:0,bad:0,bag:0,bagger:0,bagger_num_batch:0,bar:0,base:0,base_lay:0,basenam:0,basic:[],batch:0,batch_axi:0,batch_dim:0,batchnorm:0,befor:0,behav:0,behavior:0,being:0,belong:0,below:1,besid:0,between:0,bilinear:0,binar:0,binari:0,black:0,block:0,block_activ:0,block_bottleneck_compress:0,block_bottleneck_featur:0,block_bottleneck_num_step:0,block_bottleneck_num_unit:0,block_conv_kernel_s:0,block_conv_num_featur:0,block_growth_r:0,blue:0,bodi:[],bool:0,both:0,bottleneck:0,bottom:0,bound:0,box:0,bracket:0,branch:0,build:0,c1t:0,c2t:0,cach:0,calcul:0,call:0,callabl:0,callback:0,can:[0,1],carlo:0,carri:0,categorical_crossentropi:0,central:0,certain:0,chain:0,chainabl:0,chang:0,channel:0,channelwis:0,check:0,check_dataset:0,checkpoint:0,child:0,class_threshold:0,class_weight:0,classif:0,classifi:0,clean:0,clean_metadata:0,clean_sign:0,collect:0,collinear:0,color:0,column:0,column_norm:0,column_standard:0,columnwis:0,combin:0,common:0,compact:0,compar:0,compil:0,compile_and_train:0,compile_model:0,complet:0,compon:0,compress:0,comput:0,concat:0,concaten:0,conf_int:0,confid:0,confidence_interv:0,conflict:0,connect:0,consequ:0,conserv:0,consid:0,consider:0,consist:0,consol:0,constant:0,construct:0,constructor:0,contain:0,content:0,contigu:0,continu:[0,1],control:0,conv2d:0,conv_leg:0,conveni:0,convent:0,convert:0,convolut:0,convolutiona:0,coordiant:0,coordin:0,copi:1,copynth_recurs:1,core:0,correspond:0,cram:0,creat:[0,1],create_fig:0,criteria:0,crop:0,csv:[0,1],current:0,curv:0,custom:0,d_time_edge_ind:0,d_time_threshold:0,daili:1,daily_to_cont:1,data:[0,1],data_:0,databas:[0,1],datafram:0,datapoint:0,dataset:0,dataset_typ:0,date:0,datestr:0,datestr_to_timestamp:0,datetim:0,datetime_end:0,datetime_format:0,datetime_start:0,datetimeindex:0,db_script:1,dcnn:0,dcnn_filt24_kernel9x9_pexp135:0,dcnn_upsampl:0,decreas:0,deep:0,defin:0,delet:0,delta:0,denot:0,dens:0,dense_net:0,densenet:0,depend:0,describ:0,descript:0,destandard:0,detail:0,determin:0,deviat:0,dict:0,dictat:0,dictionari:0,differ:0,differenti:0,digit:0,dimens:0,dimension:0,direct:0,discard:0,discrimin:1,disjoint:0,disk:0,displac:0,displai:0,distanc:0,distribut:0,divid:0,docstr:0,doe:0,dot:0,down:0,draw:0,drop:0,drop_tabl:0,dropout:0,due:0,dummi:0,each:0,earlier:0,edg:0,effect:0,either:0,element:0,elementin:0,elu:0,encod:0,end:0,endogen:0,endogenous_test:0,endogenous_train:0,engin:0,ensembl:0,ensemble_statist:0,ensemblepredict:0,entir:0,entri:0,epoch:0,error:0,estim:0,estimate_ci:0,estimate_image_standard:0,estimate_process_standard:0,etc:0,evalu:0,even:0,everi:0,exact:0,exampl:0,except:0,excerpt:1,exclud:0,execut:0,exhaust:0,exist:0,exogen:0,exogenous_test:0,exogenous_train:0,expans:0,explicitli:0,explor:[0,1],explore_sign:0,expon:0,extend:0,extens:0,extract:0,extract_slag_sign:[0,1],extractor:0,fact:0,factor:0,fals:0,fcn:0,featur:0,feature_reduction_mlp:0,fed:0,feed:0,field:0,figur:0,filanam:0,file:0,file_path:0,filenam:0,filename_to_timestamp:0,fill:0,filter:0,filter_imag:1,filter_metadata:0,find:0,find_contiguous_tim:0,first:0,fit:0,fix:0,flatten:0,flush:0,fly:0,folder:[0,1],folder_path:0,follow:0,font:0,fontsiz:0,form:0,format:0,format_str:0,forward:0,found:0,four:0,fraction:0,frame:0,framework:0,from:[0,1],from_recip:0,frozen:0,full:0,fulli:0,function_1:0,function_2:0,function_from_recipe_lin:0,further:0,gener:[],generate_in_memory_process_data_arrai:1,geometr:0,get:[0,1],get_frame_numb:0,get_idg:0,get_indic:0,get_process_area:0,get_valid_index_rang:0,give:0,given:0,global:0,global_aggr_mlp:0,good:0,good_signal_tag:0,gradient:0,gradient_sensitivity_analysi:0,graph:0,green:0,grei:0,grid_activ:0,grow:0,growth:0,guid:1,hand:0,handl:0,handler:0,hard:0,hardcod:0,has:0,have:0,head:[],height:0,height_origin:0,height_shift_rang:0,help:[0,1],helper:0,here:[0,1],hidden:0,high:0,higher:0,highest:0,highlight:0,histori:0,history_length:0,historysampl:0,hold:0,hole:0,horizont:0,hot:0,how:0,howev:0,huang:0,hungri:0,identifi:0,identity_reducer_funct:0,illeg:0,imag:[0,1],image_channel:0,image_path:0,imageaugmentingprotocol:0,imageclassificationxyaug:0,imagedata:0,imagedatagener:0,imagesegmentationxyaug:0,imoverlai:0,implement:0,includ:0,incom:0,increas:0,ind_featur:0,independ:0,index:[0,1],index_high:0,index_low:0,indic:0,individu:0,infinit:0,inform:0,inherit:0,init:0,initi:0,inp:0,input:0,input_:0,input_lay:0,input_spac:0,insert:0,insert_background_class:0,instanc:0,instanti:0,instead:0,integ:0,interact:0,interest:0,interim:0,interim_lay:0,interpol:0,interpret:0,interv:0,intial:0,introduc:0,invalid:0,invari:0,invis:0,invok:0,irregular:0,is_random:0,is_signal_name_valid:0,island:0,iter:0,its:0,itself:0,jan:0,jpg:0,json:0,just:0,justifi:0,keep:0,kei:0,kera:0,kernel:0,kernel_s:0,kernel_shap:0,keyword:0,kk4:1,kth:0,kwarg:0,label:0,label_channel:0,laid:0,lambda:0,last:0,latent:0,later:0,layer:0,learn:0,learnabl:0,least:0,legaci:0,len:0,length:0,less:0,letter:0,level:0,like:0,line:0,linear:0,linewidth:0,link:0,linked_rate_plot:0,list:0,list_folder_cont:0,list_tabl:0,lkcamadm01:1,load:0,load_image_data:0,local:0,log1l:0,log:0,logist:0,longer:0,look:0,lookback:0,loop:0,loos:0,loss:0,low:0,lower:0,lowest:0,machin:0,made:0,mai:1,main:0,maintain:0,make:0,manag:0,mani:0,manipul:0,manner:0,map:0,mask:0,mask_imag:0,masked_imag:0,maskimag:0,match:0,matplotlib:0,matric:0,matrix:0,max:0,max_:0,maximum:0,maxpool:0,mean:0,mean_norm:0,meant:0,meet:0,member:0,memori:0,memoriless_engineeredstat:0,memoriless_mlp:0,merg:0,metadata:0,method:0,metric:0,middl:0,might:0,min:0,misc:[],misc_tensor:0,miscellan:0,mlp:0,mode:0,model:[0,1],model_1:0,model_2:0,model_ensembl:[],model_manag:[],model_mngr:0,model_path:0,modelensembl:0,modelmanag:0,modul:[0,1],mono:0,monoton:0,mont:0,more:0,most:0,mostli:0,mse:0,multi:0,multiarrai:0,multipl:0,multipli:0,must:0,n_batch:0,n_channel:0,n_data:0,n_featur:0,n_histori:0,n_imag:0,n_length:0,n_list:0,n_lookback:0,n_skip:0,n_step:0,name:0,name_tag:0,namespac:0,nan:0,narx:0,nearest:0,neck:0,neck_activ:0,neck_bottleneck_compress:0,neck_bottleneck_num_step:0,neck_bottleneck_num_unit:0,neck_conv_kernel_s:0,neck_conv_num_featur:0,need:[0,1],neg:0,neighborhood:0,neighborhood_s:0,net:0,network:[0,1],neural:0,next:0,nin1:0,nin2:0,nin:0,node:0,non:0,none:0,norm_stat:0,normal:0,normal_stat:0,normalize_column:0,normalstatist:0,note:0,num_batch:0,num_block:0,num_class:0,num_compon:0,num_epoch:0,num_featur:0,num_filt:0,num_histori:0,num_imag:0,num_pool:0,num_pooled_featur:0,num_pyramid_level:0,num_res_block:0,num_sampl:0,num_show:0,num_step:0,num_unit:0,num_upsampl:0,num_work:0,number:0,numer:0,numpi:0,object:0,observ:0,occasion:0,offer:0,on_valu:0,one:0,one_hot_encod:0,onli:0,only_filenam:0,opagu:0,open:0,oper:0,ops:0,optim:0,option:[0,1],order:[0,1],ordereddict:0,origin:0,other:0,otherwis:[0,1],out:0,output:0,output_max:0,output_min:0,output_rang:0,over:0,overlai:0,overrid:0,own:[0,1],pad:0,page:[0,1],pair:0,pan:0,panda:0,param:[],paramet:0,parameter:0,parametr:0,parent:0,parenthes:0,part:0,particular:0,pass:0,path:0,path_to_mask:0,path_to_model:0,path_to_recip:0,pattern:0,pca:0,pca_compon:0,pca_components_nam:0,pca_reconstruct:0,pca_reconstruction_nam:0,per:0,persist:0,persistentprepostfunct:0,pipelin:0,pixel:0,pixelwis:0,place:0,placehold:0,pleas:0,plot:0,plot_data:0,png:0,pool:0,pool_activ:0,pool_bottleneck_compress:0,pool_compress:0,pool_compression_num_featur:0,pool_expans:0,pool_siz:0,pool_strid:0,pool_typ:0,popul:[0,1],posit:0,post:0,postprocess:0,postprocessor:0,powershel:1,practic:0,pre:0,pre_pca:0,pre_post:[],pre_post_fn:[],pred_1:0,pred_2:0,predict:0,predictor:0,prefix:0,preprocess:0,preprocessor:0,present:0,preserv:0,press:0,previou:0,previous:0,primarili:0,print:0,prior:0,prioriti:0,priority_class:0,probabl:0,problem:0,proce:0,proceed:0,process:1,process_area:0,processdata:0,processinterpolatedinmemori:0,processor:0,processsynchedtimesampl:0,processsynchedtimewithsensit:0,processtestingprotocol:0,produc:[0,1],progress:0,project:0,project_dir:0,propag:0,properti:0,protocol:[],provid:0,pull:0,put:1,pyramid:0,pyramid_pooling_net:0,python:0,qualiti:0,queri:0,query_at:0,query_between:0,query_datetime_bound:0,query_datetime_rang:0,quick:1,rais:0,random:0,random_mod:0,random_se:0,randomized_feature_bagg:0,randomli:0,rang:0,range_norm:0,raster:0,rate:0,rate_data:0,read:0,read_imag:0,read_label:0,read_recip:0,read_sign:0,realiz:0,reason:0,recept:0,recip:0,recipe_lin:0,recipe_nam:0,recipe_path:0,reconstruct:0,rectangl:0,red:0,reduc:0,reducer_activ:0,reducer_compress:0,reducer_regressor:0,reducer_step:0,reduct:0,ref1:0,ref2:0,refactor:0,refer:0,referenc:0,reformat:0,regress:0,regular:0,rel:0,relat:0,relationship:0,relative_path:0,relev:0,relu:0,remaind:0,remov:0,render:0,repeat:0,replac:0,report:0,repres:0,represent:0,reproduc:0,request:0,requir:0,rerun:1,res_bottleneck_activ:0,res_bottleneck_num_featur:0,res_conv_activ:0,res_conv_kernel_s:0,res_conv_num_featur:0,res_upsample_activ:0,res_upsample_num_featur:0,resampl:0,rescal:0,residu:0,resnet:0,resnet_upsampl:0,respect:0,result:[0,1],revers:0,rigid:0,rotat:0,rotation_rang:0,row:0,run:[0,1],same:0,sampl:0,sampler:0,sampling_axi:0,save:[0,1],save_histori:0,save_recip:0,scalar:0,scalar_regression_mlp:0,scale0:0,scale1:0,scale2:0,scale3:0,scale:0,scatter:0,scattered_averag:0,scattered_average_r:0,scattered_moving_averag:0,scipi:0,script:[0,1],search:[0,1],second:0,secondarili:0,see:0,seed:0,seen:0,segment:[0,1],segmentation_mask:0,segnet:0,self:0,selu:0,sens:0,sensit:0,sensitivity_analysi:1,separ:[0,1],sequenc:0,sequenti:0,sequential_skip:0,seri:0,serial:0,serv:0,set:[0,1],shape:0,share:0,shift:0,should:0,show:0,show_imag:0,show_segmentation_training_data:0,show_thumbnail:0,side:0,sigmoid:0,sign:0,signal:[0,1],signal_nam:0,signal_proc:[],signatur:0,simpl:0,simpli:0,simplif:0,simultan:0,sinc:0,singl:0,singular:0,size:0,skip:0,skip_column:0,slag:[0,1],slag_sign:0,slagsignalextractor:0,slice:0,smooth:0,softmax:0,some:0,sort:0,sort_fram:0,sorting_fn:0,spatial:0,spatial_transform:0,spatialtransformaugment:0,spatiotempor:0,special:0,specif:0,specifi:0,splice:0,split:0,split_featur:0,sql:0,sql_between:0,squar:0,squeez:0,sse:0,sta:0,stack:0,stai:0,standard:0,standardize_column:0,start:[0,1],start_index:0,start_indic:0,stat:0,state:0,statist:0,statu:0,std:0,stem:0,stem_activ:0,stem_bottleneck_compress:0,stem_bottleneck_num_step:0,stem_bottleneck_num_unit:0,stem_conv_kernel_s:0,stem_conv_num_featur:0,stem_featur:0,stem_num_featur:0,stem_num_res_block:0,stem_pool:0,stem_res_bottleneck_activ:0,stem_res_bottleneck_num_featur:0,stem_res_conv_activ:0,stem_res_conv_kernel_s:0,stem_res_conv_num_featur:0,step:0,still:0,store:0,str:0,strategi:0,stride:0,string:0,structur:0,subfold:0,submodul:0,subsampl:0,subsequ:0,subset:0,subtract:0,subtract_mean:0,suggest:1,sum:0,sum_valu:0,summari:0,summat:0,support:0,suppos:0,suppress:0,sure:0,synch:0,syntact:0,tabl:0,table_of_scalar:0,tableofscalar:0,tabular:0,tag:0,take:0,take_axi:0,take_channel:0,take_featur:0,taken:0,target:0,task:0,tell:0,tempor:[0,1],temporal_activ:0,temporal_compress:0,temporal_reduction_conv:0,temporal_s:0,temporal_step:0,temporal_strid:0,tensor:0,tensorflow:0,term:0,test:[],test_images_imag:0,test_images_text:0,test_reducer_funct:0,test_sequence_text:0,testingprotocol:0,text:0,than:0,thei:0,them:0,therefor:0,thi:[0,1],third:0,those:0,thread:0,three:0,threshold:0,thu:0,thumbnail:0,tick:0,time:0,timestamp:0,timestamp_to_datetim:0,timestep:0,timestr:0,timestring_to_datetim:0,titl:0,told:0,top:0,topolog:[],total:0,traceabl:0,train:1,train_discriminator_network:1,train_process_model:1,train_segmenter_network:1,trainin:0,trainingprotocol:0,transform:0,transformed_imag:0,translat:0,transpar:0,transpos:0,treat:0,tupl:0,two:0,type:0,typic:0,uncertainti:[],unchang:0,under:0,understand:0,unimport:0,union:0,unit:0,uniti:0,unix:0,unless:1,unlik:0,updat:0,upon:0,upper:0,upsampl:0,upsample_output_expans:0,upsampling_activ:0,upsampling_res_conv_kernel_s:0,upsampling_res_upsample_activ:0,upsampling_res_upsample_num_featur:0,usag:0,use:0,used:0,useful:0,uses:0,using:[0,1],usual:0,utc:0,util:[],valid:0,validation_split:0,valu:0,valueerror:0,vanilla:0,variabl:0,varianc:0,version:0,vertic:0,visual:[0,1],visualize_sign:0,volum:0,wai:0,want:0,weight:0,well:0,were:0,what:0,whatev:0,when:0,where:0,wherev:0,whether:0,which:0,white:0,width:0,width_origin:0,width_shift_rang:0,window:0,window_s:0,wise:0,within:0,without:0,work:0,worker:0,workflow:0,wrapper:0,write:0,write_imag:0,written:0,x_batch:0,x_folder:0,x_format_str:0,xticklabel:0,y_batch:0,y_folder:0,y_format_str:0,yield:0,ylabel:0,you:[0,1],your:[0,1],zero:0,zeroth:0,zoom:0,zoom_rang:0},titles:["Documentation for the Code","Welcome to lkab_slag_ai\u2019s documentation!"],titleterms:{"abstract":0,basic:0,bodi:0,code:0,db_handl:0,document:[0,1],gener:0,head:0,indic:1,lkab_slag_ai:1,misc:0,model_ensembl:0,model_manag:0,pre_post:0,pre_post_fn:0,process:0,protocol:0,signal_proc:0,tabl:1,test:0,todo:0,topolog:0,train:0,uncertainti:0,util:0,welcom:1}})