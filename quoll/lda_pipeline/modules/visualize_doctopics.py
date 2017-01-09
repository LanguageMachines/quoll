
import glob
from luiginlp.engine import Task, WorkflowComponent, InputFormat, registercomponent, InputSlot, Parameter, BoolParameter
from quoll.lda_pipeline.functions.lda_visualizer import LDAVisualizer

class Task_Visualizedoctopics(Task):
 
    in_ldadir = InputSlot()
    in_corpusdir = InputSlot()

    group_documents = Parameter()
    #topic_labels = Parameter()
    select_topics = Parameter()
    doctopics = BoolParameter()
    probs_by_topic = BoolParameter()
    heatmap = BoolParameter()    
    topic_words = BoolParameter()
    network_graph = BoolParameter()
        
    def out_visualizations(self):
        return self.outputfrominput(inputformat='ldadir', stripextension='.topics.outputdir', addextension='.visualizations')
                            
    def run(self):
        # setup output directory
        self.setup_output_dir(self.out_visualizations().path)

        # set lda model path
        lda_model = self.in_ldadir().path + '/model.pcl'
        
        # set lda dictionary path
        lda_dict = self.in_corpusdir().path + '/corpus.dict'
        
        # set list of documents
        documents = [filename for filename in glob.glob(self.in_corpusdir().path + '/*.corpus.txt')]
        document_names = [filename.split('/')[-1].strip('.corpus.txt') for filename in documents]        
        
        # set topic labels
#        print('TOPIC LABELS',self.topic_labels,type(self.topic_labels))
#        if not self.topic_labels == 'False':
#            with open(self.topic_labels,'r',encoding='utf-8') as tl_in:
#                topic_labels = tl_in.read().strip().split('\n')
        
        #    topic_labels = False
        
        
        
        # set lda_vis object
        lda_vis = LDAVisualizer(lda_model, lda_dict)
        lda_vis.set_document_names(document_names)
        lda_vis.generate_documents_topics_raw(documents)
        lda_vis.set_fig_values()

        if not self.group_documents == 'False':
            # set group_documents dictionary
            document_group = {}
            with open(self.group_documents) as gd_in:
                group_documents_list = gd_in.read().strip().split('\n')
            for line in group_documents_list:
                kv = line.split('\t')
                document_group[kv[1]] = kv[0]
            lda_vis.set_groups(document_group)
        else:
            document_group = False

        if self.network_graph:
            # plot network of documents/groups
            network_fn = self.out_visualizations().path + '/network'
            lda_vis.visualize_network_graph(network_fn)

#        if self.select_topics != 'False':
 #           indices = [int(x) for x in self.select_topics.split(',')]
        #    lda_vis.set_topics(indices)
        #    lda_vis.set_topic_labels(indices)
        lda_vis.set_topic_labels()

        
        if self.doctopics:
            # make stacked bar plot of topics by document 
            stacked_bar_fn = self.out_visualizations().path + '/doctopic_probs.png'
            lda_vis.visualize_document_topic_probs(stacked_bar_fn)

        if self.probs_by_topic:
            # plot document probabilities by topic
            standard_topic_bar_fn = self.out_visualizations().path + '/document_probs'
            lda_vis.visualize_document_probs_bytopic(standard_topic_bar_fn)

        if self.heatmap:
            # plot document-topic heatmap
            heatmap_fn = self.out_visualizations().path + '/heatmap.png'
            if self.select_topics != 'False':
                indices = [int(x) for x in self.select_topics.split(',')]               
                lda_vis.visualize_document_topics_heatmap(heatmap_fn,indices)
            else:
                lda_vis.visualize_document_topics_heatmap(heatmap_fn)

        if self.topic_words:
            # plot most important words by topic
            topicwords_fn = self.out_visualizations().path + '/wordcloudinput'
            topicrows_fn = self.out_visualizations().path + '/topicrows.txt'
            lda_vis.topics2wordle_input(topicwords_fn)
            if self.select_topics != 'False':
                indices = [int(x) for x in self.select_topics.split(',')]
            else:
                indices = range(lda_vis.columns)
            lda_vis.topics2rows(topicrows_fn,indices)

            
                                   
                    
@registercomponent
class Component_Visualizedoctopics(WorkflowComponent):
    
    ldadir = Parameter()
    corpusdir = Parameter()

    group_documents = Parameter(default=False)
#    topic_labels = Parameter(default=False)
    select_topics = Parameter(default=False)
    doctopics = BoolParameter()
    probs_by_topic = BoolParameter()
    heatmap = BoolParameter()    
    topic_words = BoolParameter()
    network_graph = BoolParameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='ldadir',extension='.outputdir',inputparameter='ldadir'), InputFormat(self,format_id='corpusdir',extension='.corpusdir',inputparameter='corpusdir') ) ]

    def setup(self, workflow, input_feeds):
        visualizer = workflow.new_task('visualize_doctopics',Task_Visualizedoctopics,autopass=False,group_documents=self.group_documents,select_topics=self.select_topics,doctopics=self.doctopics,probs_by_topic=self.probs_by_topic,heatmap=self.heatmap,topic_words=self.topic_words,network_graph=self.network_graph)
        visualizer.in_ldadir = input_feeds['ldadir']
        visualizer.in_corpusdir = input_feeds['corpusdir']
        return visualizer
