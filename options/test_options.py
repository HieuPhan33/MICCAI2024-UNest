from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--no_save', action='store_true', default=False, help='save test images')
        parser.add_argument('--num_test', type=int, default=0, help='how many test images to run')
        parser.add_argument('--half', action='store_true', default=False, help='first half data')
        parser.add_argument('--half_data', default=False,  action='store_true', help='Halve size the dataset')

        # transformer parameters
        # parser.add_argument('--ngf_cytran', type=int, default=16, help='number of down')
        # parser.add_argument('--n_downsampling', type=int, default=3, help='number of down')
        # parser.add_argument('--depth', type=int, default=3, help='number of down')
        # parser.add_argument('--heads', type=int, default=6, help='number of down')
        # parser.add_argument('--dropout', type=float, default=0.05, help='number of down')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        # parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
