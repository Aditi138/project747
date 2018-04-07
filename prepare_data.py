from dataloaders.dataloader import DataLoader
import argparse
import sys
from line_profiler import LineProfiler

if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf8')
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train", help="[train,load] load: pre-processing data and storing in pickle format"
                                                                  " train: assumes data already stored in pickle format")
    

    parser.add_argument("--input_folder", type=str, default=None,
                        help="Path to the directory of raw documents")
    parser.add_argument("--summary_path", type=str, default=None,
                        help="Path to the summary file")
    parser.add_argument("--qap_path", type=str, default=None,
                        help="Path to the question file")                        
    parser.add_argument("--document_path", type=str, default=None,
                        help="Path to the document information file")                        
    parser.add_argument("--pickle_folder", type=str, default=None,
                        help="Path to the target folder for output")   

    parser.add_argument("--small_number", type=int, default=-1,
                        help="Pickle small number of documents for testing purposes")                                                
    


    args = parser.parse_args()
    loader=DataLoader(args)

    # profiler=LineProfiler()
    # profile_function=profiler(loader.process_data)
    # profile_function(args.input_folder, args.summary_path, args.qap_path, args.document_path, args.pickle_folder, small_number=args.small_number)
    # profiler.print_stats()

    loader.process_data(args.input_folder, args.summary_path, args.qap_path, args.document_path, args.pickle_folder, small_number=args.small_number)
