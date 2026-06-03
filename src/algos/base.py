import pickle


class BaseAlgorithm:
    def __init__(
        self,
        baselearner_fn,
        baselearner_args,
        optim_fn,
        T,
        T_val,
        T_test,
        train_batch_size,
        test_batch_size,
        lr,
        device,
        batching_eps,
        test_adam,
        operator=max,
        **kwargs
    ):
        """Initialization of the meta-learning algorithm

        Parameters
        ----------
        baselearner_fn: constructor function

        baselearner_args: dict

        optim_fn: constructor function

        T: int

        T_val: int

        T_test: int

        lr: float

        device: str

        batching_eps: bool

        test_adam: bool
            Optimize weights with Adam, LR = 0.001 at test time.
        operator: function = max
            Objective function. In case of RMSE, it is a minimization objective (min function),
            in case of accuracy the maximization objective (max function)

        """

        # Constructor function for the baselearner
        self.baselearner_fn = baselearner_fn
        # Keyword arguments for base-learner
        self.baselearner_args = baselearner_args
        # Constructor function for the optimizer to use
        self.optim_fn = optim_fn
        # Number of update steps to parameters per task
        self.T = T
        # Number of weight updates at validation time
        self.T_val = T_val
        # Number of weight updates at test time
        self.T_test = T_test
        # Training batch size
        self.train_batch_size = train_batch_size
        # Test batch size
        self.test_batch_size = test_batch_size
        # Learning rate for (meta-)optimizer
        self.lr = lr
        # Device to run model operations on
        self.device = device
        self.trainable = True
        self.episodic = True
        # Batching from episodic data
        self.batching_eps = batching_eps

        self.test_adam = test_adam
        self.operator = operator

    def train(self, train_x, train_y, test_x, test_y):
        raise NotImplementedError()

    def val(self, train_x, train_y, test_x, test_y):
        raise NotImplementedError()

    def dump_state(self):
        raise NotImplementedError()

    def load_state(self, state):
        raise NotImplementedError()

    def store_file(self, filename):
        state = self.dump_state()

        with open(filename, "wb+") as f:
            pickle.dump(state, f)

    def read_file(self, filename, **kwargs):
        with open(filename, "rb") as f:
            state = pickle.load(f)

        self.load_state(state, **kwargs)
