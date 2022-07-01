
from random import sample, shuffle
from collections import Counter
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Aliens:
    def __init__(self, att_list):
        #Store list of 48 attributes
        self.att_list = att_list
        #List of all aliens
        self.alien_list = []
        #Dict with training set, renewed each generation
        self.training_dict = {}
        #Dict with test set, renewed each generation
        self.test_dict = {}
        #Dict with original alien-att combinations
        self.alien_attribute = {}
        #List with all unique attributes per generation
        self.unique_att = []
        #length of unique attributes per generation
        self.att_length = []
        #Accuracy score per generation
        self.accuracy_score = []
        #Dict with attributes per alien from training
        self.fd = {}
        #Structure scores
        self.structure_scores_zero = []
        self.structure_scores_one = []
        self.structure_scores_two = []

        

    def build(self, color, shape, movement):
        #Build alien from 3 features and store as tuple and create empty dict with all aliens
        for i in color:
            for j in shape:
                for k in movement:
                    self.alien_list.append((i,j,k))
                    self.fd[(i,j,k)] = []
        return self.alien_list

    def combine(self):
        #Iterate over all aliens and assign 6 random attributes and store as tuple in a dict
        for alien in self.alien_list:
            random_att = (sample(self.att_list, 6))
            self.alien_attribute[alien] = (random_att)
        
        #Store number of attributes and structure scores of Gen 0
        uniques = self.evaluate(self.alien_attribute)
        zero, one, two = self.zscores(self.alien_attribute, uniques)
        self.structure_scores_zero.append(zero)
        self.structure_scores_one.append(one)
        self.structure_scores_two.append(two)
        return self.alien_attribute

    def remember(self, formed_alien):
        #Shuffle the alien-attribute dict and select 13 random aliens for training set and the others as test set
        temp_list = list(formed_alien.items())
        shuffle(temp_list)
        formed_alien = dict(temp_list)

        #Make copies of current generation for later comparison
        alien_copy = formed_alien.copy()
        formed_alien_train = dict(temp_list[:13])
        train_copy = formed_alien_train.copy()
        formed_alien_test = dict(temp_list[13:])
        test_copy = formed_alien_test.copy()

        #Iterate over aliens from training set and remember 2 random attribute 3 times
        N = 2
        for key in formed_alien_train.keys():
            #Create attributes to choose from
            pick_from = formed_alien_train[key]
            #Remember 3 times and add to empty alien dict
            for _ in range(3):
                self.fd[key].append(sample(pick_from, N))
            
        #Merge list of attribute lists per alien
        for k, v in self.fd.items():
            self.fd[k] = list(itertools.chain(*v))

        #Reproduction phase, loop over all aliens
        for key in formed_alien.keys():
            attributes = []
            train_al = False
            #If alien from training set is encountered
            if self.fd[key] != []:
                #Bool value indicating whether alien from training set is encountered
                train_al = True
                for _ in range(6):
                    #If all attributes remembered from training phase are already sampled, go to next step 
                    if all(elem in attributes for elem in self.fd[key]):
                        break

                    #If still attributes from training phase in memory, sample from that
                    else:
                        sample1 = str(sample(self.fd[key], 1))[2:-2]
                        #Check whether attribute has already been chosen
                        while sample1 in attributes:
                            sample1 = str(sample(self.fd[key], 1))[2:-2]
                    #Add sample to attribute list
                    attributes.append(sample1)
            
            #If enough attributes sampled
            if len(attributes) == 6:
                #Add 6 new attributes to the alien and update the training set for measuring accuracy later on
                formed_alien[key] = attributes.copy()
                formed_alien_train[key] = attributes.copy()
                attributes.clear()
            
            #If not enough attributes
            else:
                temp_attributes = []
                #Look for aliens in training set with 2 shared features and append their attributes to a list
                for training_key in self.fd.keys():
                    if training_key != key:
                        if (key[0] == training_key[0] and key[1] == training_key[1]) or (key[0] == training_key[0] and key[2] == training_key[2]) or (key[1] == training_key[1] and key[2] == training_key[2]):
                            temp_attributes.append(self.fd[training_key])
                #Merge list of lists
                temp_attributes = list(itertools.chain(*temp_attributes))

                #Sample attributes for as long as possible
                if set(temp_attributes):
                    for _ in range(6 - len(attributes)):
                        if all(elem in attributes for elem in temp_attributes):
                            break
                        else:
                            sample1 = str(sample(temp_attributes, 1))[2:-2]
                            while sample1 in attributes:
                                sample1 = str(sample(temp_attributes, 1))[2:-2]
                        attributes.append(sample1)

                #If enough attributes
                if len(attributes) == 6:
                    #Add 6 new attributes to the alien and update the training and test set for measuring accuracy later on
                    formed_alien[key] = attributes.copy()
                    if train_al:
                        formed_alien_train[key] = attributes.copy()
                    else:
                        formed_alien_test[key] = attributes.copy()
                    attributes.clear()

                else:
                    temp_attributes.clear()
                    #Look for aliens with 1 shared feature from training set and remember their attributes
                    for training_key in self.fd.keys():
                        if training_key != key:
                            if (key[0] == training_key[0] or key[1] == training_key[1] or key[2] == training_key[2]):
                                temp_attributes.append(self.fd[training_key])
                    #Merge list of lists
                    temp_attributes = list(itertools.chain(*attributes))

                    #Fill list of attributes with samples from attributes found in aliens with 1 shared feature or add random
                    for _ in range(6 - len(attributes)):
                        #Add random attributes if no attributes left to sample from
                        if all(elem in attributes for elem in temp_attributes) or bool(temp_attributes):
                            sample2 = str(sample(self.att_list, 1))[2:-2]
                            while sample2 in attributes:
                                sample2 = str(sample(self.att_list, 1))[2:-2]
                            attributes.append(sample2)

                        #sample from attributes of aliens with 1 shared feature
                        else:   
                            sample2 = sample(temp_attributes, 1)
                            while sample2 in attributes:
                                sample2 = sample(temp_attributes, 1)
                            attributes.append(sample2)

                    #Add new attributes to alien and the test set alien for measuring accuracy later on and clear lists for next alien
                    temp_attributes.clear()
                    formed_alien[key] = attributes.copy()
                    if train_al:
                        formed_alien_train[key] = attributes.copy()
                    else:
                        formed_alien_test[key] = attributes.copy()
                    attributes.clear()

        #Clear dict from training phase for next generation
        for value in self.fd.values():
            del value[:]

        #evaluate the total number of attributes and calculate structure scores
        uniques = self.evaluate(formed_alien)
        zero, one, two = self.zscores(formed_alien, uniques)

        #Store structure scores
        self.structure_scores_zero.append(zero)
        self.structure_scores_one.append(one)
        self.structure_scores_two.append(two)

        #Calculate accuracy
        training_accuracy = self.train_accuracy(train_copy, formed_alien_train)
        test_accuracy = self.train_accuracy(test_copy, formed_alien_test)
        full_accuracy = self.train_accuracy(alien_copy, formed_alien)
   
        return formed_alien, training_accuracy, test_accuracy, full_accuracy

    def evaluate(self, gen_results):
        #Count total number of attributes
        total_att = list(gen_results.values())
        total_att = list(itertools.chain(*total_att))
        total_att = set(total_att)

        #append both all unique elemenents and the amount of unique elements in seperate lists
        self.unique_att.append(total_att)
        self.att_length.append(len(total_att))
        return len(total_att)
    
    def train_accuracy(self, first_gen, second_gen):
        length = []
        #Count number of attributes that appear in gen i and gen i+1 in the same alien
        for (_,v),(_,v2) in zip(first_gen.items(),second_gen.items()):
            x = list(set(v).intersection(v2))
            length.append(len(x))
        #Calculate percent of aliens reproduced correctly from previous generation
        return np.round((sum(length)/(6*len(length))*100))

    def zscores(self, generation, uniques):
        #Calculate z-scores
        uniques = uniques - 6
        overlap_zero = []
        overlap_one = []
        overlap_two = []

        #Load monte carlo data
        means = pd.read_json(r'C:\users\VINCE\OneDrive\Documenten\Studie\Scriptie\dict_mean.json', typ = 'series')
        sd = pd.read_json(r'C:\Users\VINCE\OneDrive\Documenten\Studie\Scriptie\dict_std.json', typ = 'series')
        df = pd.concat([means, sd], axis = 1)
        df.columns = ['mean', 'std']
    
        #Calculate overlap
        for key, value in generation.items():
            for key2, value2 in generation.items():
                if key != key2:
                    overlap = len(set(value).intersection(set(value2)))
                    if (key[0] == key2[0] and key[1] == key2[1]) or (key[0] == key2[0] and key[2] == key2[2]) or (key[1] == key2[1] and key[2] == key2[2]):
                        overlap_two.append(overlap)
                    elif (key[0] == key2[0] or key[1] == key2[1] or key[2] == key2[2]):
                        overlap_one.append(overlap)
                    else:
                        overlap_zero.append(overlap)
            
        #Raw structure scores
        raw_zero = np.mean(overlap_zero)
        raw_one = np.mean(overlap_one)
        raw_two = np.mean(overlap_two)

        #Simulated scores
        monte_scores = df.iloc[uniques]
        monte_mean = monte_scores["mean"]
        monte_sd = monte_scores["std"]

        #Final structure scores
        struc_score_zero = np.round((raw_zero - monte_mean) / monte_sd, 6)
        struc_score_one = np.round((raw_one - monte_mean) / monte_sd, 6)
        struc_score_two = np.round((raw_two - monte_mean) / monte_sd, 6)

        return struc_score_zero, struc_score_one, struc_score_two
    

if __name__ == '__main__':  
    #Make lists of features and attributes and lists to store the evaluation and accuracy scores
    accuracy_scores = []
    evaluation_scores = []
    training_score = []
    test_score = []
    struc_list0 = []
    struc_list1 = []
    struc_list2 = []
    
    color_list = ['red','blue','green']
    shape_list = ['square','circle','triangle']
    movement_list = ['straight','bounce','turn']
    att_list_gen0 =  ['Demanding','Thoughtful','Keen', 'Happy','Disagreeable','Simple','Fancy','Plain', 'Excited','Studious','Inventive',
        'Creative','Thrilling','Intelligent', 'Proud', 'Daring', 'Bright', 'Serious', 'Funny', 'Humorous', 'Sad', 'Lazy', 'Dreamer',
        'Helpful', 'Simple-minded', 'Friendly', 'Adventurous', 'Timid', 'Shy', 'Pitiful', 'Cooperative', 'Lovable', 'Ambitious', 'Quiet',
        'Curious', 'Reserved', 'Pleasing', 'Bossy', 'Witty', 'Energetic', 'Cheerful', 'Smart', 'Impulsive', 'Humorous', 'Sad', 'Lazy', 'Dreamer',
        'Helpful']

    #Do 40 chains of the experiment
    for _ in range(40):
        #List to store alien-att associations per generation
        all_aliens = []
        train_aliens = []
        test_aliens = []
        all_acc_scores = []
        chain = Aliens(att_list_gen0)

        #Build aliens
        chain.build(color_list, shape_list, movement_list)

        #Form alien-attribute associations for generation 0 and store them
        gen0 = chain.combine()
        all_aliens.append(dict.copy(gen0))

        #Iterate 7 times
        for _ in range(7):
            gen0, gentrain, gentest, total_acc = chain.remember(gen0)
            all_aliens.append(dict.copy(gen0))
            train_aliens.append(gentrain)
            test_aliens.append(gentest)
            all_acc_scores.append(total_acc)
            

        #Append al results of chain to list
        evaluation_scores.append(chain.att_length)
        struc_list0.append(chain.structure_scores_zero)
        struc_list1.append(chain.structure_scores_one)
        struc_list2.append(chain.structure_scores_two)
        training_score.append(train_aliens)
        test_score.append(test_aliens)
        accuracy_scores.append(all_acc_scores)

    
    #Plotting results
    plot_eval_list = []
    plot_acc_list = []
    plot_train_list = []
    plot_test_list = []
    plot_struc_list0 = []
    plot_struc_list1 = []
    plot_struc_list2 = []
    for gen in range(7):
        plot_acc_list.append(np.mean([item[gen] for item in accuracy_scores]))
        plot_train_list.append(np.mean([item[gen] for item in training_score]))
        plot_test_list.append(np.mean([item[gen] for item in test_score]))
    for gen in range(8):
        plot_eval_list.append(np.mean([item[gen] for item in evaluation_scores]))
        plot_struc_list0.append(np.mean([item[gen] for item in struc_list0]))
        plot_struc_list1.append(np.mean([item[gen] for item in struc_list1]))
        plot_struc_list2.append(np.mean([item[gen] for item in struc_list2]))

    
    #Mean attributes
    plt.plot(plot_eval_list)
    plt.xlabel("Generation")
    plt.ylabel("Mean total attributes")
    plt.ylim(0,48)
    plt.title("Mean number of total attributes used for 40 chains")
    plt.show()

    #Mean accuracies 
    plt.plot([1,2,3,4,5,6,7], plot_train_list, label = 'Seen aliens')
    plt.scatter(x=range(1, 8), y=plot_train_list, s=5)
    plt.plot([1,2,3,4,5,6,7], plot_test_list, label = 'Unseen aliens')
    plt.scatter(x=range(1, 8), y=plot_test_list, s=5)
    plt.ylim(0,100)
    plt.xlabel("Generation")
    plt.ylabel("Mean accuracy (%)")
    plt.legend()
    plt.show()

    #Mean structure
    plt.plot(plot_struc_list0)
    plt.scatter(x=range(8), y=plot_struc_list0, s=5)
    plt.plot(plot_struc_list1)
    plt.scatter(x=range(8), y=plot_struc_list1, s=5)
    plt.plot(plot_struc_list2)
    plt.scatter(x=range(8), y=plot_struc_list2, s=5)
    plt.axhline(y = 1.96, linestyle = '--')
    plt.xlabel("Generation")
    plt.ylabel("Mean structure (z-score)")
    plt.legend(['Zero shared features','One shared feature','Two shared features', 'Chance'])
    plt.show()