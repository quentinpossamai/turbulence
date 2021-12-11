Source code for the paper "Learning nonlinear state-space models using autoencoders" By Masti and Bemporad.

The entry point for the program is "main.py", while "reproducibility.sh" 
and "batchRun.sh" are utility script to perform test reported in the paper.

The script "generateDataset.m" can be used to generate the .mat files needed to 
perform external dataset tests (assuming that matlab system identification 
toolbox is correctly installed and the silverbox dataset files are present in the path) 

To generate the whole set of results, you can run the script "reproducibility.sh".
Such script will create in the sub-dir results (which has to be created first!) a series of
text files that contain the results of each simulation. 

The files "results*.ods" in the sub-directory "results" can be used to compute the 
statistics reported in the paper from the individual test files. There, each sheet 
corresponds to a set of tests and reports the synthetic results of each run (the data are
extracted using the bash command next to the results of each test)

The program will generate various .mat dumps that can be used to check the performance
of the NLARX from SysID toolbox using the script "NLARXcheck.m"

The results have been obtained using python 3.6.9, and keras 2.3.1, tf 1.14.0. 
Results may vary if other versions of those libraries are used.
We note that results vary between runs due to the use of multicore support of Keras
that naturally acts as indetermination source. See https://stackoverflow.com/a/52897216 ,
 https://github.com/keras-team/keras/issues/2280 and https://github.com/keras-team/keras/issues/12800

Please note that the hammerstein-wiener system class is called "LinearSystem" due to historical reasons as the the I/O non linearities were a later addition.
Also, the \Sigma_RH system is sometimes referred as "Magneto" 
Feed-forward and quasi-LPV configurations are referred as, respectively, "non-affine" and "affine" into the code/results


Please note that both the source code and the scripts perform additional benchmarks that were not 
included in the draft due to space constraints.


