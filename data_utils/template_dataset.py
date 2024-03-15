from data.base_dataset import BaseDataset, get_transform

class TemplateDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        self.image_paths = []  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        path = 'temp'    # needs to be a string
        data_A = None    # needs to be a tensor
        data_B = None    # needs to be a tensor
        return {'data_A': data_A, 'data_B': data_B, 'path': path}

    def __len__(self):
        return len(self.image_paths)
