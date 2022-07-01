
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
        #Dict with current alien-att combinations 
        self.alien_attribute = {}
        #List with all unique attributes per generation
        self.unique_att = []
        #length of unique attributes per generation
        self.att_length = []
        #Accuracy score per generation
        self.accuracy_score = []
        #Dict with attributes per feature
        self.fd = {}
        #Structure scores
        self.structure_scores_zero = []
        self.structure_scores_one = []
        self.structure_scores_two = []
        
        self.structure_scores_col = []
        self.structure_scores_shape = []
        self.structure_scores_move = []
               
    def build(self, color, shape, movement):
        #Build alien from 3 features and store as tuple and create dict with each feature
        for i in color:
            self.fd[i] = []
            for j in shape:
                self.fd[j] = []
                for k in movement:
                    self.fd[k] = []
                    self.alien_list.append((i,j,k))
        return self.alien_list

    def combine(self):
        #Iterate over all aliens and assign 6 random attributes and store as tuple in a dict
        for i in self.alien_list:
            random_att = (sample(self.att_list, 6))
            self.alien_attribute[i] = (random_att)

        # Calculate results of Gen 0
        uniques = self.evaluate(self.alien_attribute)
        zero, one, two, col, shape, move = self.zscores(self.alien_attribute, uniques)

        self.structure_scores_zero.append(zero)
        self.structure_scores_one.append(one)
        self.structure_scores_two.append(two)

        self.structure_scores_col.append(col)
        self.structure_scores_shape.append(shape)
        self.structure_scores_move.append(move)

        return self.alien_attribute

    def remember(self, formed_alien, N):
        #Select 13 random aliens for training set from current generation of aliens
        temp_list = list(formed_alien.items())
        shuffle(temp_list)
        formed_alien = dict(temp_list)

        #Copy current dict to compare later
        alien_copy = formed_alien.copy()
        formed_alien_train = dict(temp_list[:13])
        train_copy = formed_alien_train.copy()
        formed_alien_test = dict(temp_list[13:])
        test_copy = formed_alien_test.copy()

        #Set probabilites per category feature
        keyset = [0, 1, 2]
        key_probabilities = [6.75, 2.41, 3.05]
        key_norm = [float(i)/sum(key_probabilities) for i in key_probabilities]

        #Iterate over aliens and remember N amount of random attributes 3 times per feature and store in dict
        for key in formed_alien_train.keys():
            pick_from = formed_alien_train[key]
            for _ in range(3):
                keuze = str(np.random.choice(keyset, 1, p = key_norm))[1:-1]
                self.fd[key[int(keuze)]].append(sample(pick_from, N))
        
        #Merge value lists together and keep duplicates. If a feature is empty, add 6 random attrbiutes
        for k, v in self.fd.items():
            self.fd[k] = list(itertools.chain(*v))
            if self.fd[k] == []:
                self.fd[k] = (sample(self.att_list, 6))
        
        #Loop over all aliens and add attributes based on feature_dict 
        for key in formed_alien.keys():
            #Store new attributes for the alien
            attributes = []
            for _ in range(6):
                #Check if it is possible to generate new attributes from training set (in other words: if not all attributes have been picked)
                keuze = str(np.random.choice(keyset, 1, p = key_norm))[1:-1]
                if all(elem in attributes for elem in self.fd[key[0]]) and all(elem in attributes for elem in self.fd[key[1]]) and all(elem in attributes for elem in self.fd[key[2]]):
                    sample1 = str(sample(self.att_list, 1))[2:-2]
                    while sample1 in attributes:
                        sample1 = str(sample(self.att_list, 1))[2:-2]

                #Select first attribute  
                else:
                    while all(elem in attributes for elem in self.fd[key[int(keuze)]]):
                        keuze = str(np.random.choice(keyset, 1, p = key_norm))[1:-1]
                    sample1 = str(sample(self.fd[key[int(keuze)]], 1))[2:-2]
                    #If it has already been picked sample again
                    while sample1 in attributes:
                        sample1 = str(sample(self.fd[key[int(keuze)]], 1))[2:-2]
                attributes.append(sample1)
               
            #Add a copy to the alien dictionary as the new value for the key alien
            formed_alien[key] = attributes.copy()
            if key in formed_alien_train:
                formed_alien_train[key] = attributes.copy()
            else:
                formed_alien_test[key] = attributes.copy()
            attributes.clear()  

        #Clear the training values
        for value in self.fd.values():
            del value[:]
        
        #Calculate number of attributes, accuracy and structure scores
        uniques = self.evaluate(formed_alien)
        zero, one, two, col, shape, move = self.zscores(formed_alien, uniques)

        self.structure_scores_zero.append(zero)
        self.structure_scores_one.append(one)
        self.structure_scores_two.append(two)

        self.structure_scores_col.append(col)
        self.structure_scores_shape.append(shape)
        self.structure_scores_move.append(move)

        training_accuracy = self.train_accuracy(train_copy, formed_alien_train)
        test_accuracy = self.train_accuracy(test_copy, formed_alien_test)
        full_accuracy = self.train_accuracy(alien_copy, formed_alien)

        return formed_alien, training_accuracy, test_accuracy, full_accuracy

    def evaluate(self, gen_results):
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

        overlap_color = []
        overlap_shape = []
        overlap_move = []

        #Simulated data
        means = pd.read_json(r'C:\users\VINCE\OneDrive\Documenten\Studie\Scriptie\dict_mean.json', typ = 'series')
        sd = pd.read_json(r'C:\Users\VINCE\OneDrive\Documenten\Studie\Scriptie\dict_std.json', typ = 'series')
        df = pd.concat([means, sd], axis = 1)
        df.columns = ['mean', 'std']
    
        #Calculate overlap
        for key, value in generation.items():
            for key2, value2 in generation.items():
                if key != key2:
                    overlap = len(set(value).intersection(set(value2)))
                    if key[0] == key2[0]:
                        overlap_color.append(overlap)
                    if key[1] == key2[1]:
                        overlap_shape.append(overlap)
                    if key[2] == key2[2]:
                        overlap_move.append(overlap)
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

        raw_color = np.mean(overlap_color)
        raw_shape = np.mean(overlap_shape)
        raw_move = np.mean(overlap_move)

        #Simulated scores
        monte_scores = df.iloc[uniques]
        monte_mean = monte_scores["mean"]
        monte_sd = monte_scores["std"]

        #Final scores
        struc_score_zero = np.round((raw_zero - monte_mean) / monte_sd, 6)
        struc_score_one = np.round((raw_one - monte_mean) / monte_sd, 6)
        struc_score_two = np.round((raw_two - monte_mean) / monte_sd, 6)

        struc_score_color = np.round((raw_color - monte_mean) / monte_sd, 6)
        struc_score_shape = np.round((raw_shape - monte_mean) / monte_sd, 6)
        struc_score_move = np.round((raw_move - monte_mean) / monte_sd, 6)

        return struc_score_zero, struc_score_one, struc_score_two, struc_score_color, struc_score_shape, struc_score_move
    


if __name__ == '__main__':  
    #Make lists of features and attributes and lists to store the evaluation structure and accuracy scores
    accuracy_scores = []
    training_score = []
    test_score = []
    evaluation_scores =[]

    struc_list0 = []
    struc_list1 = []
    struc_list2 = []

    struc_list_col = []
    struc_list_shape = []
    struc_list_move = []
    
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

        #Iterate 7 times down the chain and save accuracy scores
        for _ in range(7):
            gen0, gentrain, gentest, total_acc = chain.remember(gen0, 3)
            all_aliens.append(dict.copy(gen0))
            train_aliens.append(gentrain)
            test_aliens.append(gentest)
            all_acc_scores.append(total_acc)

        #Measure accuracy, total att. and structure scores
        evaluation_scores.append(chain.att_length)
        training_score.append(train_aliens)
        test_score.append(test_aliens)
        accuracy_scores.append(all_acc_scores)

        struc_list0.append(chain.structure_scores_zero)
        struc_list1.append(chain.structure_scores_one)
        struc_list2.append(chain.structure_scores_two)

        struc_list_col.append(chain.structure_scores_col)
        struc_list_shape.append(chain.structure_scores_shape)
        struc_list_move.append(chain.structure_scores_move)


    
    #Plotting results
    plot_eval_list = []
    plot_acc_list = []
    plot_train_list = []
    plot_test_list = []

    plot_struc_list0 = []
    plot_struc_list1 = []
    plot_struc_list2 = []

    plot_struc_list_col = []
    plot_struc_list_shape = []
    plot_struc_list_move = []

    for gen in range(7):
        plot_acc_list.append(np.mean([item[gen] for item in accuracy_scores]))
        plot_train_list.append(np.mean([item[gen] for item in training_score]))
        plot_test_list.append(np.mean([item[gen] for item in test_score]))
    for gen in range(8):
        plot_eval_list.append(np.mean([item[gen] for item in evaluation_scores]))
        plot_struc_list0.append(np.mean([item[gen] for item in struc_list0]))
        plot_struc_list1.append(np.mean([item[gen] for item in struc_list1]))
        plot_struc_list2.append(np.mean([item[gen] for item in struc_list2]))

        plot_struc_list_col.append(np.mean([item[gen] for item in struc_list_col]))
        plot_struc_list_shape.append(np.mean([item[gen] for item in struc_list_shape]))
        plot_struc_list_move.append(np.mean([item[gen] for item in struc_list_move]))

    #Plot mean number of attributes
    plt.plot(plot_eval_list)
    plt.xlabel("Generation")
    plt.ylabel("Mean total attributes")
    plt.ylim(0,48)
    plt.title("Mean number of total attributes used for 40 chains")
    plt.show()


    #Plot accuracy
    plt.plot([1,2,3,4,5,6,7], plot_train_list, label = 'Seen aliens')
    plt.scatter(x=range(1, 8), y=plot_train_list, s=5)
    plt.plot([1,2,3,4,5,6,7], plot_test_list, label = 'Unseen aliens')
    plt.scatter(x=range(1, 8), y=plot_test_list, s=5)
    plt.ylim(0,100)
    plt.xlabel("Generation")
    plt.ylabel("Mean accuracy (%)")
    plt.legend()
    plt.show()

    #Plot structure scores
    plt.plot(plot_struc_list0)
    plt.scatter(x=range(8), y=plot_struc_list0, s=5)
    plt.plot(plot_struc_list1)
    plt.scatter(x=range(8), y=plot_struc_list1, s=5)
    plt.plot(plot_struc_list2)
    plt.scatter(x=range(8), y=plot_struc_list2, s=5)
    plt.axhline(y = 1.96, linestyle = '--')
    plt.xlabel("Generation")
    plt.ylabel("Mean structure (z-score)")
    plt.legend(['Zero shared features','One shared feature','Two shared features', 'Chance'], loc = 'lower right', fontsize = 'small')
    plt.show()

    plt.plot(plot_struc_list_col)
    plt.scatter(x=range(8), y=plot_struc_list_col, s=5)
    plt.plot(plot_struc_list_shape)
    plt.scatter(x=range(8), y=plot_struc_list_move, s=5)
    plt.plot(plot_struc_list_move)
    plt.scatter(x=range(8), y=plot_struc_list_shape, s=5)
    plt.axhline(y = 1.96, linestyle = '--')
    plt.xlabel("Generation")
    plt.ylabel("Mean structure (z-scores)")
    plt.legend(['Color','Shape','Movement', 'Chance'], loc = 'lower right')
    plt.show()
    

