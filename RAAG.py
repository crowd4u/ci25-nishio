### import
#import pdb; pdb.set_trace()
import csv
import numpy as np
import sys
import os
import re

"""
Function: em()
    Run the EM estimator on the data from the Dawid-Skene paper
"""
### em
def em(responses, responses_evi, filename):
    # run EM
    sample_classes, error_rates, samples, observers, classes, evidences, evi_weight , evidence_reliability= run(responses, responses_evi, filename)

    return sample_classes, error_rates, samples, observers, classes, evidences, evi_weight, evidence_reliability

"""
Function: dawid_skene()
    Run the Dawid-Skene estimator on response data
Input:
    responses: a dictionary object of responses:
        {tasks: {observers: [labels]}}
    tol: tolerance required for convergence of EM
    max_iter: maximum number of iterations of EM
"""    
### run
def run(responses, responses_evi, filename, tol=0.00001, max_iter=50, init='average'):
    # convert responses to counts
    (samples, observers, classes, evidences, counts) = responses_to_counts(responses, responses_evi)
    print("num samples:", len(samples)) 
    print("Observers:", observers) 
    print("Classes:", classes) 
    print("Evidences:", evidences) 
    
    # initialize
    iter = 0
    converged = False
    old_class_marginals = None
    old_error_rates = None
    old_evi_weight = None
    sample_classes = initialize(counts)
    print("Iter\tlog-likelihood\tdelta-CM\tdelta-ER")
    
    # while not converged do:
    while not converged:
        iter += 1
        
        # M-step
        (class_marginals, error_rates, evi_weight, evidence_reliability) = m_step(counts, sample_classes)

        # E-setp
        sample_classes = e_step(counts, class_marginals, error_rates, evi_weight, evidence_reliability)

        # check likelihood
        log_L = calc_likelihood(counts, class_marginals, error_rates, sample_classes)

        # check for convergence
        if old_class_marginals is not None:
            class_marginals_diff = np.sum(np.abs(class_marginals - old_class_marginals))
            error_rates_diff = np.sum(np.abs(error_rates - old_error_rates)) # type: ignore
            evi_weight_diff = np.sum(np.abs(evi_weight - old_evi_weight)) # type: ignore
            print(iter, '\t', log_L, '\t%.6f\t%.6f' % (class_marginals_diff, error_rates_diff))
            if (class_marginals_diff < tol and error_rates_diff < tol and evi_weight_diff < tol) or iter > max_iter:
                converged = True
        else:
            print(iter, '\t', log_L)

        # update current values
        old_class_marginals = class_marginals
        old_error_rates = error_rates
        old_evi_weight = evi_weight
        
    
    np.set_printoptions(precision=2, suppress=True)
    answer_labels = np.argmax(sample_classes, axis = 1) # retrieves the index of the class with the highest probability for each task.
    estimate_labels = []
    for answer in answer_labels:
        estimate_labels.append(classes[answer])
    score = 0
    for i in range(len(correctset)):
        if str(correctset[i]) == estimate_labels[i]:
            score += 1
    # Print final results
    print("---score---")
    print(score/len(samples))

    return sample_classes, error_rates, samples, observers, classes, evidences, evi_weight, evidence_reliability

"""
Function: responses_to_counts()
    Convert a matrix of annotations to count data
Inputs:
    responses: dictionary of responses {tasks:{workers:[responses]}}
    responses_evi: dictionary of evidences {tasks:{workers:[evidences]}}
Return:
    samples: list of tasks 
    observers: list of observers 
    classes: list of possible task classes 
    evidences: list of evidences
    counts of the number of times each responses were received 
        by each observer from each task: [tasks, observers, classes, evidences] 
"""    
### responses to counts
def responses_to_counts(responses, responses_evi):
    samples = list(responses)
    p = re.compile(r'\d+')
    samples.sort(key=lambda s: int(p.search(s).group())) # type: ignore
    nsamples = len(samples)

    # determine the observers and classes
    observers = set()
    classes = set()
    evidences = set()
    for i in samples:
        i_observers = list(responses[i])
        for k in i_observers:
            if k not in observers:
                observers.add(k)
            ik_responses = responses[i][k]
            
            ik_responses_evi = responses_evi[i][k]
            classes.update(ik_responses)
            evidences.update(ik_responses_evi)

    classes = list(classes)
    classes.sort()
    nClasses = len(classes)

    observers = list(observers)
    observers.sort(key=lambda s: int(p.search(s).group())) # type: ignore
    nObservers = len(observers)
    
    evidences = list(evidences)
    evidences.sort()
    nEvidences = len(evidences)

    # create a 3d array to hold counts
    counts = np.zeros([nsamples, nObservers, nClasses, nEvidences])

    # convert responses to counts
    for sample in samples:
        i = samples.index(sample)
        for observer in list(responses[sample]):
            k = observers.index(observer)
            for response in responses[sample][observer]:
                j = classes.index(response)
                for response_evi in responses_evi[sample][observer]:
                    j_e = evidences.index(response_evi)
                    counts[i,k,j,j_e] += 1
    sample_class = np.zeros([nsamples,nClasses])
    for p in range(nsamples):
        counts_sum = np.sum(counts[p,:,:,:], axis=0)
        sample_class[p,:] = np.sum(counts_sum, axis=1)
    
    return (samples, observers, classes, evidences, counts)

"""
Function: initialize()
    Get initial estimates for the true task classes using counts
    see equation 3.1 in Dawid-Skene (1979)
Input:
    counts of the number of times each responses were received 
        by each observer from each task: [tasks, observers, classes, evidences] 
Returns:
    task_classes: matrix of estimates of true task classes:
        [tasks, responses]
"""  
### initialize(majority voting)
def initialize(counts):
    [nsamples, nObservers, nClasses, nEvidences] = np.shape(counts)
    # sum over observers
    response_sums = np.sum(counts,1)
    # create an empty array
    sample_classes = np.zeros([nsamples, nClasses])
    # for each sample, take the average number of observations in each class
    for p in range(nsamples):
        counts_sum = np.sum(counts[p,:,:,:], axis=0)
        sample_classes[p,:] = np.sum(counts_sum, axis=1)
        sample_classes[p,:] /= np.sum(sample_classes[p,:],dtype=float)

    return sample_classes

"""
Function: m_step()
    e-stemp of EM algorithm
Input: 
    counts of the number of times each responses were received 
        by each observer from each task: [tasks, observers, classes, evidences] 
    sample_classes: Matrix of current assignments of tasks to classes: [tasks, classes]
Returns:
    class_marginal: the class marginals [tasks, classes]
    error_rates: worker confusion matrices, [workers, classes, classes]
    evi_weight: generation probabilities of evidences, [classes, evidences]
    evidence_reliability: strength of evidences, [tasks, evidences]
"""
def m_step(counts, sample_classes):
    [nsamples, nObservers, nClasses, nEvidences] = np.shape(counts)
    counts_evi = np.sum(counts, axis=1)
    counts_ds = np.sum(counts, axis=3)
    p_ey = np.sum(np.sum(counts,axis = 0), axis=0) 
    evi_weight = np.zeros([nClasses, nEvidences])

    # compute class marginals p(t)
    class_marginals = np.sum(sample_classes,0) / float(nsamples)
    
    #evi_weight p(e|t)
    for i in range(nClasses):
        evi_weight[i,:] = p_ey[i] 
        evi_weight[i,:] = evi_weight[i,:] / np.sum(evi_weight[i,:])
    
    # compute error_rates 
    error_rates = np.zeros([nObservers, nClasses, nClasses])
    
    # compute evidence reliability 
    evidence_reliability = np.zeros([nsamples, nEvidences])
    for k in range(nObservers):
        for j in range(nClasses):
            for l in range(nClasses):
                error_rates[k, j, l] = np.dot(sample_classes[:,j], counts_ds[:,k,l])
            #error_rates 
            sum_over_responses = np.sum(error_rates[k,j,:])
            if sum_over_responses > 0:
                error_rates[k,j,:] = error_rates[k,j,:] / float(sum_over_responses)
                
    #evidence reliability
    for k in range(nsamples):           
        for j in range(nEvidences):
            if sum(counts_evi[k,:,j]) != 0.0:
                evidence_reliability[k][j] = np.dot(sample_classes[k,:], counts_evi[k,:,j])/sum(counts_evi[k,:,j])
            else:
                evidence_reliability[k][j] = 0
        sum_over_evidence_re = np.sum(evidence_reliability[k,:])
        if sum_over_evidence_re > 0:
            evidence_reliability[k,:] = evidence_reliability[k,:] / float(sum_over_evidence_re)
                
    return (class_marginals, error_rates, evi_weight, evidence_reliability)

""" 
Function: e_step()
    m-stemp of EM algorithm
Inputs:
    counts: counts of the number of times each responses were received 
        by each observer from each task: [tasks, observers, classes, evidences] 
    class_marginal: the class marginals [tasks, classes]
    error_rates: worker confusion matrices, [workers, classes, classes]
    evi_weight: generation probabilities of evidences, [classes, evidences]
    evidence_reliability: strength of evidences, [tasks, evidences]
Returns:
    sample_classes: Soft assignments of tasks to classes
        [tasks, classes]
"""      
def e_step(counts, class_marginals, error_rates, evi_weight, evidence_reliability):
    [nsamples, nObservers, nClasses, nEvidences] = np.shape(counts) 
    sample_classes = np.zeros([nsamples, nClasses])
    counts_evi = np.sum(counts, axis=1)
    counts_ds = np.sum(counts, axis=3)
    
    for i in range(nsamples):
        for j in range(nClasses):
            estimate = class_marginals[j] #p(t)
            estimate *= (1-np.prod(np.power(1-evi_weight[j,:],counts_evi[i,j,:]))) #p(e|t)
            estimate *= np.prod(np.power(error_rates[:,j,:], counts_ds[i,:,:])) #p(y|e,t)
            sample_classes[i,j] = estimate * (1 - np.prod(np.power(1-evidence_reliability[i,:],counts_evi[i,j,:]))) #p(y|e,t)
        sample_sum = np.sum(sample_classes[i,:])
        if sample_sum > 0:
            sample_classes[i,:] = sample_classes[i,:] / float(sample_sum)
            
    return sample_classes

"""
Function: calc_likelihood()
    Calculate the likelihood given the current parameter estimates
    This should go up monotonically as EM proceeds
Inputs:
    counts: counts of the number of times each responses were received 
        by each observer from each task: [tasks, observers, classes, evidences] 
    class_marginal: the class marginals [tasks, classes]
    error_rates: worker confusion matrices, [workers, classes, classes]
    sample_classes: Soft assignments of tasks to classes
        [tasks, classes]
Returns:
    Likelihood given current parameter estimates
"""  
### calculate likelihood
def calc_likelihood(counts, class_marginals, error_rates, sample_classes):
    [nsamples, nObservers, nClasses, nEvidences] = np.shape(counts)
    log_L = 0.0

    for i in range(nsamples):

        sample_likelihood = 0.0

        for j in range(nClasses):
            class_prior = class_marginals[j]
            sample_likelihood += sample_classes[i,j]
        temp = log_L + np.log(sample_likelihood)

        if np.isnan(temp) or np.isinf(temp):
            sys.exit()

        log_L = temp

    return log_L

### input data
def input_data():
    # loads the input data file
    filename = sys.argv[1]
    
    responses = {}
    responses_evi = {}
    # The contents of responses/responses_evi are structured as follows.
    # responses = {TaskID : {WorkerID : [answer/evidences], WorkerID : [answer/evidences], ........}, 
    #              TaskID : {WorkerID : [answer/evidences], WorkerID : [answer/evidences], ........},
    #             }

    # reads the data file and creates the dictionaries responses and responses_evi.
    with open(filename, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        global correctset
        correct = []
        for row in reader:
            if row[0] in responses.keys():
                responses[row[0]][row[1]] = [row[2]]
                responses_evi[row[0]][row[1]] = [row[3]]
                redundancy += 1
            else:
                redundancy = 1
                responses[row[0]] = {}
                responses[row[0]][row[1]] = [row[2]]
                responses_evi[row[0]] = {}
                responses_evi[row[0]][row[1]] = [row[3]]
            correct.append(row[4])
            correctset = correct[::redundancy]

        
    return responses, responses_evi, filename

### outut data
def write_file(sample_classes, error_rates, samples, observers, classes, evidences, filename):
    answer_labels = np.argmax(sample_classes, axis = 1) # # retrieves the index of the class with the highest probability for each task.
    estimate_labels = []
    for answer in answer_labels:
        estimate_labels.append(classes[answer])

    # outputs the estimated labels for each task.
    class_file = 'class_' + filename
    with open(class_file, "w", newline="") as f:
        writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['TaskID', 'Estimate_label']) # header
        for num in range(len(samples)):
            writer.writerow([samples[num], estimate_labels[num]])

    
    # outputs the estimated ability of each worker.
    error_file = 'error_' + filename
    with open(error_file, "w", newline="") as f:
        writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # header
        header = ['WorkerID']
        for j in classes:
            for l in classes:
                header.append(j+'_'+l)
        writer.writerow(header)
        
        for k in range(len(observers)):
            row = []
            row.append(observers[k])
            for j in range(len(classes)):
                for l in range(len(classes)):
                    row.append(error_rates[k,j,l])
            writer.writerow(row)
            

### main
def main():
    responses, responses_evi, filename = input_data() # データの入力
    sample_classes, error_rates, samples, observers, classes, evidences, evi_weight, evidence_reliability = em(responses, responses_evi, filename) # RAAG (EM algorithm)
    write_file(sample_classes, error_rates, samples, observers, classes, evidences, filename) # ファイルを出力

if __name__ == '__main__':
    main() 
