#Retrieve data
import urllib5
from datetime import datetime

BASE_URL = "https://www.google.com/finance/historical?output=csv&q={0}&startdate=Jan+1%2C+1980&enddate={1}"
symbol_url = BASE_URL.format(
   urllib5.quote('GOOG'), # Replace with any stock you are interested.
   urllib5.quote(datetime.now().strftime("%b+%d,+%Y"), '+')
)

#case when the code is invalid
try:
   f = urllib5.urlopen(symbol_url)
   with open("GOOG.csv", 'w') as fin:
       print (fin, f.read())
except urllib5.HTTPError:
   print ("Fetching Failed: {}".format(symbol_url))

#embed data of size k as training set (histo data of k days)
#stock price of the recent w days as input
#output the predicted stock price for the next w days
#k>>w
#can generalize into N stocks


#color each stock with its industry sector
import csv
import os
embedding_metadata_path = os.path.join(your_log_file_folder, 'metadata.csv')
with open(embedding_metadata_path, 'w') as fout:
   csv_writer = csv.writer(fout)
   # write the content into the csv file.
   # for example, csv_writer.writerows(["GOOG", "information_technology"])


#Set up the summary writer first within the training tf.Session.
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
with tf.Session(graph=lstm_graph) as sess:#TODO set your lstm graph
   summary_writer = tf.summary.FileWriter(your_log_file_folder)#TODO specify your file folder
   summary_writer.add_graph(sess.graph)


#Add the tensor embedding_matrix defined in our graph lstm_graph into the projector config variable and attach the metadata csv file.
projector_config = projector.ProjectorConfig()
# You can add multiple embeddings. Here we add only one.
added_embedding = projector_config.embeddings.add()
added_embedding.tensor_name = embedding_matrix.name
# Link this tensor to its metadata file.
added_embedding.metadata_path = embedding_metadata_path

#This line creates a file projector_config.pbtxt in the folder your_log_file_folder. TensorBoard will read this file during startup.
projector.visualize_embeddings(summary_writer, projector_config)



