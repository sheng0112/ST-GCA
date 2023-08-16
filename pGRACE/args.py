import argparse


def init_args():
    parser = argparse.ArgumentParser(description='Graph Contrastive Learning with Adaptive Augmentation')

    # Experiment Settings
    parser.add_argument('--basePath', type=str, default="D:\PycharmProjects\DeepST-main\data\DLPFC")
    parser.add_argument('--seed', type=int, default=3407)  ###
    parser.add_argument('--learning_rate', type=float, default=0.01)  ###

    # Model Design
    parser.add_argument('--num_hidden', type=int, default=128)  ###
    parser.add_argument('--num_proj_hidden', type=int, default=64)  ###
    parser.add_argument('--activation', type=str, default='prelu')  ###
    parser.add_argument('--base_model', type=str, default='GCNConv')  ###

    # Training Hyperparameters
    parser.add_argument('--num_epochs', type=int, default=20)  ###
    parser.add_argument('--drop_edge_rate_1', type=float, default=0.2)  ###
    parser.add_argument('--drop_edge_rate_2', type=float, default=0.6)  ###
    parser.add_argument('--drop_feature_rate_1', type=float, default=0.4)
    parser.add_argument('--drop_feature_rate_2', type=float, default=0.3)

    # Model Hyperparameters
    parser.add_argument('--tau', type=float, default=0.1)  ###
    parser.add_argument('--weight_decay', type=float, default=1e-5)  ###

    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')  ###
    parser.add_argument('--batch_size_I', '-bI', type=int, default=128, help='Batch size for spot image data')  ###
    parser.add_argument('--current_epoch_I', '-curEI', type=int, default=0, help='current epoches for spot image data')
    parser.add_argument('--max_epoch_I', '-meI', type=int, default=500, help='Max epoches for spot image data')
    parser.add_argument('--latent_I', '-lI', type=int, default=128,
                        help='Feature dim for latent vector for spot image data')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--test_prop', default=0.05, type=float, help='the proportion data for testing')
    parser.add_argument('--tillingPath', '-TP', type=str, default=None, help='image data directory')
    parser.add_argument('--image_size', '-iS', type=int, default=32, help='image size for spot image data')
    parser.add_argument('--lr_I', type=float, default=0.001, help='Learning rate for spot image data')

    parser.add_argument('--path', type=str, default="/opt/data/private")  ###
    # parser.add_argument('--path', type=str, default="D:\PycharmProjects\DeepST-main\data\DLPFC")
    parser.add_argument("--gene_preprocess", choices=("pca", "hvg"), default="hvg")  ###
    parser.add_argument("--n_gene", choices=(3000, 1000), default=3000)  ###
    parser.add_argument('--img_size', type=int, default=16)  ###
    parser.add_argument('--name', type=str, default="151673")  ###
    parser.add_argument('--batch_size', type=int, default=128)

    return parser.parse_args()
