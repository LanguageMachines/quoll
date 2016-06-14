#iris, script to run mallet on ponyland with different parameters 
#output  options are:

#output-topic-keys :file with the topics and their overall weight  (shown with most important words) 
#doc-topics : file that shows for every document those topics that exceed the TOPICTHRESHOLD 
#inferencer : mallet program to assign the created topics to a new data set

DIR=/vol/customopt/machine-learning/src/mallet/mallet

cd $DIR


MDIR=$1
FILENAME=$2
INPUTDIR=$3
NUMTOPICS=$4

INTERVAL=10
TOPICTHRESHOLD=0.1
RANDVAR=123456
#$RANDOM-> gives a random number
NUMITER=2000



#ALPHA=0.1 not need to specify when using the optimize-interval
#BETA=0.01


## convert files for mallet#
nice bin/mallet import-dir --keep-sequence --input $INPUTDIR --output $MDIR/$FILENAME


$DIR/bin/mallet train-topics --num-iterations $NUMITER --random-seed  $RANDVAR --num-topics $NUMTOPICS --optimize-interval $INTERVAL --input $MDIR/$FILENAME --doc-topics-threshold $TOPICTHRESHOLD  --output-doc-topics  $MDIR/$FILENAME.$n.NT$NUMTOPICS.I$INTERVAL.THR$TOPICTHRESHOLD.doc-topics --output-topic-keys $MDIR/$FILENAME.NT$NUMTOPICS.I$INTERVAL.rand$RANDVAR.topic_keys.txt --inferencer-filename $MDIR/$FILENAME.NT$NUMTOPICS.I$INTERVAL.rand$RANDVAR.inferencer 



