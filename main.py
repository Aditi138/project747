from dataloaders.dataloader import DataLoader
import argparse
import sys

if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf8')
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default="/../narrativeqa/out/summary/train.pickle")
    parser.add_argument("--valid_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--pretrain_path", type=str, default=None, help="Path to the pre-trained word embeddings")

    args = parser.parse_args()
    loader = DataLoader(args)

    train_documents = loader.load_documents( args.train_path, summary_path=None)
    valid_documents = loader.load_documents(args.valid_path, summary_path=None)
    test_documents = loader.load_documents(args.test_path, summary_path=None)