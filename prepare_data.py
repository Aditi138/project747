from dataloaders.dataloader import DataLoader
import argparse
import sys


if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf8')
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default=None,
                        help="[train,load]")
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
    parser.add_argument("--summary_only", action="store_true", help="create summary pickles")
    parser.add_argument("--small_number", type=int, default=-1,
                        help="Pickle small number of documents for testing purposes")                                                

    
    args = parser.parse_args()
    loader=DataLoader(args)


    loader.process_data(args.input_folder, args.summary_path, args.qap_path, args.document_path, args.pickle_folder, small_number=args.small_number, summary_only=args.summary_only)




    # profiler=LineProfiler()
    # profile_function=profiler(loader.process_data)
    # profile_function(args.input_folder, args.summary_path, args.qap_path, args.document_path, args.pickle_folder, small_number=args.small_number)
    # profiler.print_stats()