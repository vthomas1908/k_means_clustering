


#run as python3 k_means.py <trainingSamples> <testingSamples> <k> <writeFile>
#where k = the k number of clusters to use
#and writeFile is the file to write data to get imaging

import sys
import os
import math
import random
import copy

#function to put data from file into working list
def format_data(data_file):
  #make sure file exists
  if not os.path.exists(data_file):
    print(data_file + " does not exist")
    sys.exit(0)

  #open file
  text_file = open(data_file)

  #create list of lines
  data = text_file.readlines()

  #close the file
  text_file.close()

  #put data in integer form instead of string
  for line in range(len(data)):
    new = []

    for val in data[line].split(","):
      if val != "\n":
        new.append(int(val))

    data[line] = new

  return data


#class for making the clusters
class Model(object):
  def __init__(self, data, k):
    #note: data = data to train on
    self.data = data
    self.k = k
    self.attributes = 64
    self.digits = 10
    self.size = len(data)
    self.centers = [] 
    self.points = []
    self.mse = []
    self.avg_mse = 0
    self.mss = 0
    self.avg_entropy = 0

    #after training, associate cluster w/ digit
    self.k_class = []

    #test_data info
    self.correct = 0
    self.accuracy = 0
    #empty confusion matrix
    self.c_matrix = []
    for i in range(self.digits):
      self.c_matrix.append([])
      for j in range(self.digits):
        self.c_matrix[i].append(0)



  #initializes the centers attributes to a rand int
  #b/w 0 and 16 (inclusive)
  #rand seed is sys time by default
  def __set_init_centers(self):
    for clust in range(self.k):
      new = []
      for attr in range(self.attributes):
        new.append(random.randint(0, 16))
      self.centers.append(new)



  #calc the euclidean distance for a given set of points
  def __calc_dist(self, p1, p2):
    val = 0

    for attr in range(self.attributes):
      val += math.pow(p1[attr] - p2[attr], 2)

    val = math.sqrt(val)

    return val



  #func for getting the mean square error
  #this is the sum(dist of each center to each of its samples ^2)
  #divided by the number of samples in that cluster
  def __mean_square_error(self, dist, points):
    for k in range(self.k):
      val = 0
      length = len(points[k])

      #if no samples belong to the cluster,
      #no need to continue
      if length < 1:
        self.mse.append(0)

      else:

        #the items in points list are index to samples
        for sample_idx in points[k]:

          #don't want all distances, just the ones belonging
          #to this cluster
          #distances are euclidean, so square this value
          val += math.pow(dist[k][sample_idx], 2)

        #mean, so divide by total num of samples in the cluster
        self.mse.append(val/length)



  #func to get avg mean square error
  def __avg_mean_square_error(self):
    val = 0

    for k in range(self.k):
      val += self.mse[k]

    self.avg_mse = val/self.k



  #func to get mean square separation
  #this is done from outside the class
  def mean_square_separation(self):
    val = 0
    #get dist for each pair from the clusters and add to val
    #since dist is euclidean, square it
    for i in range(self.k - 1):
      for j in range(i + 1, self.k):
        val += math.pow(self.__calc_dist(self.centers[i], self.centers[j]), 2)

    #divide val by the total number of pairs
    pairs = (self.k * (self.k - 1))/2
    self.mss = val/pairs



  #func to get the mean entropy
  def mean_entropy(self):
    e = []
    lengths = []

    for k in range(self.k):
      val = 0
      length = len(self.points[k])
      lengths.append(length)

      for digit in range(10):
        num = 0

        #check how many training samples in cluster are the digit
        #recall class is the last value in the sample
        for sample_idx in self.points[k]:
          if self.data[sample_idx][self.attributes] == digit: 
            num += 1
        #increas this cluster's val by 
        #prob(member of cluster belongs to class (digit))
        #times log base 2 of that prob
        if num != 0 and length != 0:
          prob = num/length
          val += prob * math.log(prob, 2)

      #append the entropy for this cluster to the list
      e.append(-1 * val)

    #after all entropies found, calc mean entropy
    #sum for each cluster:
    #  (samples in cluster/total samples) * cluster entropy
    for k in range(self.k):
      self.avg_entropy += (lengths[k]/self.size) * e[k]



  #this makes the clusters, finds the avg mse
  def make_clusters(self):
    #set centers only once
    self.__set_init_centers()

    #make a flag so that we know when clusters are done
    flag = True

    #keep track of previous points so we know when to
    #update flag (when no more changes)
    previous_points = []

    #do the following until clusters are done
    while flag:
      #keep track of distances and points
      dist = []
      points = []
      for i in range(self.k):
        points.append([])

      #find distance for each sample at each center
      for k in range(self.k):
        temp = []
        for sample in self.data:
          temp.append(self.__calc_dist(sample, self.centers[k]))

        dist.append(temp)

      #the cluster with the lowest distance to a sample 
      #"owns" that sample
      for sample_idx in range(self.size):
        low = dist[0][sample_idx]
        k_idx = 0

        for k in range(1, self.k):
          if dist[k][sample_idx] < low:
            low = dist[k][sample_idx]
            k_idx = k

        points[k_idx].append(sample_idx)

      #move center points to new center
      #by find mean of each attribute for samples at each cluster
      for k in range(self.k):

        #check if cluster k owns any samples (avoid div 0)
        if points[k]:
          for attr in range(self.attributes):

            #bookkeeping
            num = 0
            val = 0

            for sample_idx in points[k]:

              #increment num (divide by number)
              #and val by attribute val for this sample in cluster
              num += 1
              val += self.data[sample_idx][attr]

            #update the center point at this attribute
            self.centers[k][attr] = val/num

      #check if old points == new points, if so -->DONE!
      if previous_points == points:
        flag = False

        #make a copy of the points so this can
        #be used for the entropy
        self.points = copy.deepcopy(points)

        #now that clusters formed
        #get mean square error and its avg
        self.__mean_square_error(dist, points)
        self.__avg_mean_square_error()

      #otherwise, keep track of the current points and
      #go again
      else:
        previous_points = copy.deepcopy(points)



  #fuction to associate clusters with most common digit
  #after clustering is done
  def associate(self):
    digits = 10

    for k in range(self.k):
      temp = [0] * 10

      #get a count of each digit in the cluster (training)
      for sample_idx in self.points[k]:
        temp[self.data[sample_idx][self.attributes]] += 1

      #the highest count wins
      high = temp[0]
      digit = 0
      for d in range(1, digits):
        if temp[d] > high:
          high = temp[d]
          digit = d

      self.k_class.append(digit)



  #function to predict classification of given test data
  #this also computes accuracy data
  def predict(self, test_data):
    for sample in test_data:
      dist = []
      
      for center in self.centers:

        #get distance of this sample to each center
        dist.append(self.__calc_dist(sample, center))

      #find the lowest distance to classify
      low = dist[0]
      clust = 0

      for k in range(1, self.k):
        if dist[k] < low:
          low = dist[k]
          clust = k

      classed = self.k_class[clust]
      actual = sample[self.attributes]

      #update the accuracy and confustion matrix
      self.c_matrix[actual][classed] += 1
      if classed == actual:
        self.correct += 1
      self.accuracy = self.correct/len(test_data)



  #function to print out the data needed
  def print_data(self):
    print("Training Data results:")
    print("Average Mean Square Error: ", self.avg_mse)
    print("Mean Square Separation: ", self.mss)
    print("Mean Entropy: ", self.avg_entropy)

    print("\n\nTesting Results:")
    print("Accuracy: ", self.accuracy)
    print("Confusion Matrix: ")
    for i in range(self.digits):
      for j in range(self.digits):
        print(self.c_matrix[i][j], end=" ") 
      print("\n")

    print("cluster class association: ", self.k_class)



#...having some issues with pip --> having to do imaging
#...seperately until I can figure out why libraries
#...won't install
#function to get data to pass to imaging application
#that will render an image of the cluster center
def write_data(cluster):
  text_file = open(sys.argv[4], "w")

  for i in range(cluster.k):
    for j in range(cluster.attributes):
      text_file.write(str(cluster.centers[i][j]) + " ")
    text_file.write("\n")

  text_file.close()



#main
runs = 5
models = []

training = format_data(sys.argv[1])
testing = format_data(sys.argv[2])

#make five cluster models
for run in range(runs):
  print("run: ", run)
  clusters = Model(training, int(sys.argv[3]))
  clusters.make_clusters()
  models.append(clusters)

#use the model with the lowest mean_square_average
low = models[0].avg_mse
idx = 0
for run in range(1, runs):
  if models[run].avg_mse < low:
    low = models[run].avg_mse
    idx = run

#on best model, find mean square separation,
#mean enropy, associate the clusters with a digit,
#and make predictions on the test samples
models[idx].mean_square_separation()
models[idx].mean_entropy()
models[idx].associate()
models[idx].predict(testing)
models[idx].print_data()

#write a new file that will be used for imaging
write_data(models[idx])
