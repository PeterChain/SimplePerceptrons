import sys
import network

LEARNING_RATE = 0.10

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
        if len(line) == 0:
            continue
        if line[0] == '#':
            continue
        file_array.append(line.split(' '))
    fp.close()
    return file_array


def write_dataset(filename, file_array):
    """Writes a file array to file
       each entry in array is a string in one line separated by space
    
    Arguments:
        filename {String} -- Name of the destination file
        file_array {Array} -- Array of entries
    """
    line = ' '.join((str(e) for e in file_array)

    fp = open(filename, 'w')
    fp.write(line)
    fp.close()


def new_weight_vector(size):
    """Returns a N size array generated with random numbers
    
    Arguments:
        size {Integer} -- Size of the array
    """
    result_array = [size]
    for e in result_array:
        e = randint(0, 100)
    return result_array



def run_test(filename):
    """Execute the simple perceptron network
       with a test file that includes the expected
       result value (that won't be entering the network)
    
    Arguments:
        filename {String} -- test case file
    """
    try:
        file_array = load_dataset(filename)
    except OSError as error:
        print(str(error))
        quit()

    # Build the input array
    input_vector = []
    for entry in file_array:
        input_vector.append(entry[:len(entry)-1])
    
    # Check that input vector is not initial
    if not input_vector:
        print("Input vector is initial")
        quit()
    
    # Determine if there is a weight array, if not initialize
    # one with random values
    try:
        weight_array = load_dataset(filename + '_w')
    except OSError:
        weight_vector_size = len(input_vector[0])
        weight_array = new_weight_vector(weight_vector_size)
    
    
    

if __name__ == '__main__':
    if sys.argv != 3:
        print("Usage: <program name> <mode> <path_to_set>")
        quit()
    
    if sys.argv[1] = 'run':
        run_case(sys.argv[2])
    elif sys.argv[1] = 'train':
        run_test(sys.argv[2])
    else:
        print("Invalid mode: select either 'train' or 'run'")
        quit()

    #weight_set = sys.argv[2] + '_w'
    