

# How to use the Tensorflow proposed code?


Download the iSEG-2017 data and place it in data folder. (Visit this link https://iseg2017.web.unc.edu/ to download the data. You need to register for the challenge.)



## You can run the proposed model :



Configure the flags according to your experiment.



### To run Training
####  From configure.py
### The directory where your  data is stored

                       flags.DEFINE_string('raw_data_dir', '.\Datasets',
			'the directory where the raw data is stored')
      
  ### The number of epochs to use for training      
      
                       flags.DEFINE_integer('train_epochs',300000,
			'the number of epochs to use for training')
			

Then run 


                                      Main.py





### To run Testing

###  change from train to predict



                       parser.add_argument('--option', dest='option', type=str, default='train',  help='actions: train or predict')
                       
                       
Then run 

                                      Main.py
				      
				      
###  For evaluation	


                                   evaluation.py
			   
###   For visualization results 


                                   visualize.py
			   
				      
