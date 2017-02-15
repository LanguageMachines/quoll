
from gensim import models, corpora
from matplotlib import pyplot
import numpy
from scipy.spatial import distance
from collections import defaultdict
import string
from matplotlib import rcParams

class LDAVisualizer:

    def __init__(self, model, dictionary):
        self.lda = models.LdaModel.load(model)
        self.dict = corpora.Dictionary.load(dictionary)
        self.document_topics_raw = []
        self.document_names = []
        self.document_group_dict = False
        self.group_names = False
        self.rows = False
        self.columns = False

    def set_document_names(self, docnames):
        self.document_names = docnames

    def set_fig_values(self):
        self.rows, self.columns = self.document_topics_raw.shape
        self.ind = numpy.arange(self.rows)
        self.barwidth = 0.5
        rcParams.update({'figure.autolayout': True})
        self.fig = pyplot.figure(figsize=(10, 7), dpi=280)

    def set_topic_labels(self,indices=False):
        self.topic_labels = []
        if indices:
            ts = indices
        else:
            ts = range(self.columns)
        for t in ts:
            words_topic = ' - '.join([self.dict[word[0]] for word in self.lda.get_topic_terms(t, topn=2)])
            topic_label = '#{}'.format(t) + ' ' + words_topic
            self.topic_labels.append(topic_label)

    def set_groups(self, document_group_dict):
        self.document_group_dict = document_group_dict
        self.group_names = []
        for document in self.document_names:
            if self.document_group_dict[document] not in self.group_names:
                self.group_names.append(self.document_group_dict[document])

    def set_topics(self, topic_indices):
        self.document_topics_raw = self.document_topics_raw[:,topic_indices]

    def return_topic_names(self,topic_indices):
        topic_names = []
        for t in topic_indices:
            words_topic = ' - '.join([self.dict[word[0]] for word in self.lda.get_topic_terms(t, topn=2)])
            topic_label = '#{}'.format(t) + ' ' + words_topic
            topic_names.append(topic_label)
        return topic_names

    def document2bow(self, document):
        with open(document,'r',encoding='utf-8') as infile:
            document_text = infile.read().split()
        document_bow = self.dict.doc2bow(document_text)
        return document_bow

    def return_document_topic_probabilities(self, document):
        document_bow = self.document2bow(document)
        topics = self.lda.get_document_topics(bow=document_bow, minimum_probability=0)
        probabilities = [topic[1] for topic in topics]
        return probabilities

    def generate_documents_topics_raw(self, documents):
        self.document_topics_raw = []
        for document in documents:
            self.document_topics_raw.append(self.return_document_topic_probabilities(document))
        self.document_topics_raw = numpy.array(self.document_topics_raw)

    def visualize_document_topic_probs(self, outfile):
        plots = []
        height_cumulative = numpy.zeros(self.rows)
        #fig = pyplot.figure(figsize=(21, 10), dpi=550)
        for column in range(self.columns):
            color = pyplot.cm.coolwarm(column/self.columns, 1)
            if column == 0:
                p = pyplot.bar(self.ind, self.document_topics_raw[:, column], self.barwidth, color=color)
            else:
                p = pyplot.bar(self.ind, self.document_topics_raw[:, column], self.barwidth, bottom=height_cumulative, color=color)
            height_cumulative += self.document_topics_raw[:, column]
            plots.append(p)
        pyplot.ylim((0, 1))
        pyplot.ylabel('Topics')
        pyplot.title('Topic distribution of CLS papers')
        pyplot.xticks(self.ind+self.barwidth/2, self.document_names, rotation='vertical', size = 10)
        pyplot.yticks(numpy.arange(0, 1, 10))
        pyplot.legend([p[0] for p in plots], self.topic_labels, bbox_to_anchor=(1, 1))
        self.fig.tight_layout()
        pyplot.savefig(outfile)

    def visualize_topic_document_probs(self, topic_index, outfile):
        probs = self.document_topics_raw[:, topic_index]
        pyplot.bar(self.ind, probs, width=self.barwidth)
        pyplot.xticks(self.ind + self.barwidth/2, self.document_names, rotation='vertical', size = 10)
        pyplot.title('Share of topic #' + str(topic_index))
        self.fig.tight_layout()
        pyplot.savefig(outfile)

    def visualize_document_probs_bytopic(self, standard_outfile):
        for topic_index in range(self.columns):
            outfile = standard_outfile + '_topic' + str(topic_index+1) + '.png'
            self.visualize_topic_document_probs(topic_index, outfile)
            pyplot.clf()

    def sort_doctopics_groups(self):
        groupname_rows = defaultdict(list)
        for i, doc in enumerate(self.document_names):
            group = self.document_group_dict[doc]
            groupname_rows[group].append(i)
        new_rows = []
        new_document_names = []
        for group in groupname_rows.keys():
            for index in groupname_rows[group]:
                new_document_names.append(self.document_names[index])
                new_rows.append(index)
        self.document_topics_raw = self.document_topics_raw[new_rows,:]
        self.document_names = new_document_names

    def visualize_document_topics_heatmap(self, outfile, set_topics=False):
        self.sort_doctopics_groups()
        doctopics_raw_hm = numpy.rot90(self.document_topics_raw)
        rows, columns = doctopics_raw_hm.shape
        rownames = self.topic_labels
        columnnames = self.document_names
        pyplot.pcolor(doctopics_raw_hm, norm=None, cmap='Blues')
        pyplot.gca().invert_yaxis()
        if self.group_names:
            ticks_groups = []
            bounds = []
            current_group = False
            start = 0
            for i,doc in enumerate(self.document_names):
                group = self.document_group_dict[doc]
                if group != current_group:
                    if i != 0:
                        bounds.append(i-1)
                        ticks_groups[start+int((i-start)/2)] = current_group
                    current_group = group
                    start=i
                ticks_groups.append('')
            ticks_groups[start+int((i-start)/2)] = current_group
            pyplot.xticks(numpy.arange(columns)+0.5,ticks_groups, fontsize=11)
            if set_topics:
                for index in set_topics:
                    pyplot.axhline(y=index)
                topic_names = self.return_topic_names(set_topics)
                pyplot.yticks(set_topics,topic_names,fontsize=8)
            else:
                pyplot.yticks(numpy.arange(rows)+0.5, rownames, fontsize=8)
            for bound in bounds:
                pyplot.axvline(x=bound)
        pyplot.colorbar(cmap='Blues')
        pyplot.savefig(outfile)
        pyplot.clf()

    def visualize_topic_words(self,topic_index,topic_topwords,fontsize_base):
        print('word_topic_input',topic_topwords)
        pyplot.subplot(1, self.columns, topic_index + 1)  # plot numbering starts with 1
        pyplot.ylim(0, len(topic_topwords) + 0.5)  # stretch the y-axis to accommodate the words
        pyplot.xticks([])  # remove x-axis markings ('ticks')
        pyplot.yticks([]) # remove y-axis markings ('ticks')
        pyplot.title('Topic #{}'.format(topic_index))
        for i, (word, share) in enumerate(topic_topwords):
            pyplot.text(0.1, len(topic_topwords)-i-0.5, word, fontsize=fontsize_base*share)

    def visualize_words_bytopic(self,outfile,num_words=25):
        topics = self.lda.show_topics(num_topics=self.columns,num_words=10000)
        word_topic = []
        probs = []
        for t in range(self.columns):
            word_topic_str = topics[t][1]
            prob_word = word_topic_str.split(' + ')
            word_prob = [(x.split('*')[1], float(x.split('*')[0])) for x in prob_word]
            word_prob_sorted = sorted(word_prob, key = lambda k : k[1], reverse=True)
            word_prob_pruned = word_prob_sorted[:num_words]
            probs.extend([x[1] for x in word_prob_pruned])
            word_topic.append(word_prob_pruned)
        fontsize_base = 100 / numpy.max(probs) # font size for word with largest share in corpus
        for t in range(self.columns):
            self.visualize_topic_words(t,word_topic[t],fontsize_base)
        self.fig.tight_layout()
        pyplot.savefig(outfile)

    def group_topic_vector(self):
        self.group_vector = {}
        group_vectors = defaultdict(list)
        for i,name in enumerate(self.document_names):
            group = self.document_group_dict[name]
            vector = self.document_topics_raw[i,:]
            group_vectors[group].append(vector)
        for group in self.group_names:
            vectors = numpy.array(group_vectors[group])
            topic_vector = []
            for topic in range(vectors.shape[1]):
                column = vectors[:,topic]
                topic_vector.append(numpy.mean(column))
            self.group_vector[group] = topic_vector

    def calculate_distance(self,vector1,vector2):
        vectordist = distance.euclidean(vector1,vector2)
        return vectordist

    def generate_group_distance_matrix(self):
        self.group_topic_vector()
        self.distance_matrix = []
        for group_row in self.group_names:
            row = []
            for group_column in self.group_names:
                dist = self.calculate_distance(self.group_vector[group_row],self.group_vector[group_column])
                row.append(dist)
            self.distance_matrix.append(row)
        self.distance_matrix = numpy.array(self.distance_matrix)

    def visualize_network_graph(self,outfile):
        self.generate_group_distance_matrix()
        graph = networkx.from_numpy_matrix(self.distance_matrix)
        pos1 = networkx.spring_layout(graph,k=0.1)
        pos2 = networkx.circular_layout(graph)
        for i,pos in enumerate([pos1,pos2]):
            out = outfile + '_' + str(i) + '.png'
            for key in pos.keys():
                position = pos[key]
                pyplot.scatter(position[0], position[1], marker=r'$ {} $'.format(self.group_names[key]), s=700, c=(position[1]/10.0, 0, 1 - position[1]/10.0), edgecolor='None')
            pyplot.tight_layout()
            pyplot.axis('equal')
            pyplot.savefig(out)
            pyplot.clf()

    def topics2wordle_input(self,std_outfile,num_words=100):
        word_topic = []
        probs = []
        for t in range(self.columns):
            word_topic = self.lda.get_topic_terms(t, topn=num_words)
            word_probs = [':'.join([self.dict[wordprob[0]],str(10000*wordprob[1])]) for wordprob in word_topic]
            topic_outfile = std_outfile + '_topic' + str(t) + '.txt'
            with open(topic_outfile,'w',encoding='utf-8') as wc_out:
                wc_out.write('\n'.join(word_probs))

    def topics2rows(self,outfile,topic_indices,num_words=25):
        topic_rows = []
        for t in topic_indices:
            words_topic = ','.join([self.dict[word[0]] for word in self.lda.get_topic_terms(t, topn=num_words)])
            topic_rows.append(str(t) + ',' + words_topic)
        with open(outfile,'w',encoding='utf-8') as out:
            out.write('\n'.join(topic_rows))

