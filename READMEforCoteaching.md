Usage: change train.combiner.type to coteaching in config file  
  
# difference  
  
Add coteaching in combiner.py  
  
Add train_for_coteaching and validate_for_coteaching in function.py  
    
Add coteaching mode in main/train.py(which needs two models,two optimizers,two schedulers)  
   
# todo  
  
forget_rate for coteaching is hardcoded as 0.2 in combiner  
  
num_gradual_epoch for coteaching is hardcoded as 10 epochs in combiner  
  
(so the forget_rate is 0 when it is 1st epoch and gradually increase to 0.2 during first 10 epochs)  

