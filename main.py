from dataloaders.dataloader import DataLoader
import argparse
import sys
import torch

if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf8')
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default="/../narrativeqa/out/summary/train.pickle")
    parser.add_argument("--valid_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--pretrain_path", type=str, default=None, help="Path to the pre-trained word embeddings")

    #Model parameters
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--embed_size", type=int, default=100)
    parser.add_argument("--cuda", action="store", type=str)
    parser.add_argument("--batch_length", type=int, default=10)

    args = parser.parse_args()

    if args.cuda and torch.cuda.is_available():
        vars(args)['use_cuda'] = True

    loader = DataLoader(args)

    train_documents = loader.load_documents( args.train_path, summary_path=None)
    valid_documents = loader.load_documents(args.valid_path, summary_path=None)
    test_documents = loader.load_documents(args.test_path, summary_path=None)

    loader.create_id_to_vocabulary()

    train_batches = loader.create_batches(train_documents,args.batch_length, type='train')
    valid_batches = loader.create_batches(valid_documents,args.batch_length, type='test')
    test_batches = loader.create_batches(test_documents,args.batch_length, type='test')