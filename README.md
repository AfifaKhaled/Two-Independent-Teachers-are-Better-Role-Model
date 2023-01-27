
### How to use the Tensorflow proposed code?


Download the iSEG-2017 data and place it in data folder. (Visit this link https://iseg2017.web.unc.edu/ to download the data. You need to register for the challenge.)



### You can run the proposed model :



Configure the flags according to your experiment.



### To run Training

                         flags.DEFINE_boolean("training", True, "True for Training ")

Then run 
                                      Main.py

This will train your model and save the best checkpoint according to your validation performance.

You can also resume training from saved checkpoint by setting the load_chkpt flag.




### To run Testing

                       flags.DEFINE_boolean("testing", True, "True for Testing ")
                       flags.DEFINE_boolean("training", False, "True for Training ")
                       
Then run 
                                      Main.py
