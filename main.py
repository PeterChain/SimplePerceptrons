import sys
from random import uniform
from network import *


def load_dataset(filename):
    """Loads the dataset into an array of inputs
       Each line is splitted at a whitespace where each block is placed 
       at a position of the array, empty lines and lines that start with 
       hash (#) will not be taken into account. The function returns a
       list of arrays
    
    Arguments:
        filename {String} -- Path to file with dataset
    """
    file_array = []

    fp = open(filename, 'r')
    for line in fp:
        line = line.replace('\n', '')
        line = line.replace('\r', '')
        if len(line) == 0:
            continue
        if line[0] == '#':
            continue
        file_array.append(map(float, line.split(' ')))
    fp.close()
    return file_array


def write_dataset(filename, file_array):
    """Writes a file array to file
       each entry in array is a string in one line separated by space
    
    Arguments:
        filename {String} -- Name of the destination file
        file_array {Array} -- Array of entries
    """
    line = ' '.join((str(e) for e in file_array))

    fp = open(filename, 'w')
    fp.write(line)
    fp.close()


def new_weight_vector(size):
    """Returns a N size array generated with random numbers
    
    Arguments:
        size {Integer} -- Size of the array
    """
    result_array = [None] * size
    for index, e in enumerate(result_array):
        result_array[index] = uniform(0, 1)
    return result_array



def run_test(filename, iterations):
    """Execute the simple perceptron network
       with a test file that includes the expected
       result value (that won't be entering the network)
    
    Arguments:
        filename {String} -- test case file
        iterations {Integer} -- number of times that a dataset will be run
    """
    error_count = 0

    try:
        input_array = load_dataset(filename)
    except EnvironmentError as error:
        print(str(error))
        quit()
    
    # Check that input vector is not initial
    if not input_array:
        print("Input vector is initial")
        quit()

    # Determine if there is a weight array, if not initialize
    # one with random values
    try:
        weight_vector = load_dataset(filename + '_w')[0]
    except EnvironmentError:
        weight_vector_size = len(input_array[0]) #Input array plus bias
        weight_vector = new_weight_vector(weight_vector_size)
    
    print("Current weight vector:")
    print(weight_vector)
    
    set = []
    for line in input_array:
        set.append( ([line[:len(line)-1], line[len(line)-1]]) )


    total_run = 0
    total_errors = 0
    for _ in range(iterations):
        print("Iteration: " + str(_))
        ret = train_network(set, weight_vector)
        weight_vector = ret['weights']
        total_run += ret['runs']
        total_errors += ret['errors']
    
    # import ipdb; ipdb.set_trace()
    percent_error = (total_errors * 100) / total_run
    
    print("Ran " + str(total_run) + " samples. With " + str(percent_error) + "% of errors")    
    write_dataset(filename + '_w', weight_vector)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: <program name> <mode> <path_to_set> [<iterations> (only for train)]")
        quit()
    
    if sys.argv[1] == 'run':
        run_case(sys.argv[2])
    elif sys.argv[1] == 'train':
        try:
            iter = int(sys.argv[3])
        except:
            iter = 1
        run_test(sys.argv[2], iter)
    else:
        print("Invalid mode: select either 'train' or 'run'")
        quit()
    