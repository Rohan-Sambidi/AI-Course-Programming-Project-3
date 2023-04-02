import argparse
import time

# Create the argument parser.
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--objtypes",
    default=1,
    type=int,
    help="No of object types")
    
parser.add_argument("--objcount",
    default=1,
    type=int,
    help="The number of objects for each type in the environment.")
    
parser.add_argument("--seed",
    default=int(time.time()),
    type=int,
    help="The random seed")
    
parser.add_argument("--alpha",
    default=0.3,
    type=float,
    help="The learning rate.")
    
parser.add_argument("--gamma",
    default=0.9,
    type=float,
    help="The discount factor.")
    
parser.add_argument("--episodes",
    default=500,
    type=int,
    help="The total number of episodes.")
    
parser.add_argument("--max-steps",
    default=500,
    type=int,
    help="The max. steps per episode.")
    

parser.add_argument("--file-name",
    default="results.csv",
    help="Store results in <file> in the project root directory.")

parser.add_argument("--env",
    default="bookWorld",
    type=str,
    metavar="[bookWorld/cafeWorld]",
    help="Choose environment b/w cafeWorld and bookWorld")

parser.add_argument("--clean",
    action='store_true',
    help="Overwrite output file")

parser.add_argument("--submit",
    action='store_true',
    help="Run submission for the epsilon-task")

def parse_args():
    """
        Parses the cmd line arguments.

        Returns
        ========
            args: Namespace
                The parsed args.

        Raises
        =======
            ArgumentError
                If the arguments do not parse correctly.
    """
    args = parser.parse_args()

    if args.env not in ["cafeWorld","bookWorld"]:
        raise ValueError("Environment can only be one of 'cafeWorld' or 'bookWorld'")

    if args.objtypes > 2:
        raise Exception('Maximum no. of objects available is: 2')

    elif args.objtypes < 1:
        raise Exception('Minimum no. of objects available is: 1')

    if args.objcount > 3:
        raise Exception('Maximum no. of objects available for each type is: 3')
        
    elif args.objcount < 1:
        raise Exception('Minimum no. of objects available for each type is: 1')

    return args
