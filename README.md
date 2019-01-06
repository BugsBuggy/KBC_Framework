# A Framework for Knowledge Base Completion with Embedding Models

## Prerequisites

* Python3

For lower versions change the invocation of ```python3 main.py``` in ```run_main.sh``` and ```grid_search.sh``` scripts

* PyTorch >= 0.4.1

The models require a Knowledge Base (KB) in the form of triples (subject, relation, object) in order to be trained and evaluated.
First run the ```download_datasets.sh``` script. It will download several datasets (KBs) and preprocess them.

## Running and Evaluating Models
Open the ```run_main.sh``` script and change the variables and flags according to your needs, for example the ```train``` and ```evaluate``` flags. The ```run_main.sh``` contains an exemplary configuration for running and evaluating the DistMult model. See the comments for the available choices of models, datasets and settings in general. Touch the config file only if you need access to more advanced settings.

You can use **PyTorch** for the following components:

**Activation function:** Choose an activation function from [torch.nn](https://pytorch.org/docs/stable/nn.html), for example set ```ent_func='Sigmoid()'```

**Optimizer:** Choose an optimizer from [torch.nn.optim](https://pytorch.org/docs/stable/optim.html), for example set ```optimizer='SGD'``` 

**Initialization methods:** Choose from [torch.nn.init](https://pytorch.org/docs/stable/nn.html#torch-nn-init), for example set ```init='uniform_({embs},-0.01,0.01)'``` 

**Loss function:** The loss function is currently in the config class in ```config.py```. You can choose from [torch.nn](https://pytorch.org/docs/stable/nn.html), for example set ```criterion = 'nn.BCEWithLogitsLoss()'``` in ```config.py```. See 'Adding and hiding variables' to move it to the run scripts.
You can also implement your [own components](#add_components).

If you made all settings, run the script ```run_main.sh```.

### Early Stopping
If you want to use early stopping, set the flag for early stopping in ```run_main.py``` or ```grid_search.py```. For early stopping the condition ```eval_freq < num_epochs``` must be satisfied. 
Set the ```patience``` variable. It determines the number of times both evaluation metrics (Mean Reciprocal Rank & Hits@10 for Entity-Ranking, Mean Average Precision & Hits@K for Entity-Pair ranking) are observed to be worse, until the training process is stopped. 

For example if you set ```patience = 0``` the framework will stop the training process after it observed both metrics to be worse metrics once. If you set ```patience = 1```, worse metrics are observed twice and so forth.
 

## Running Grid Search
Run the ```grid_search.sh``` script analogue to running a model, except setting multiple values for the values you want to include in the search space. Currently you can set the ```dimensions, lr (learning rate), lifted_reg (lifted regularization), init (initializer), l2_reg (l2-regularization), sampler```. To add more variables refer to the next section.



## Adding and Hiding Variables 
You have the option to hide and add variables from the config file to the current ```run_main.sh``` and ```grid_search.sh``` script or vice versa.
You can also create new variables. Follow this procedure in which the placeholder variables ```p1, p2``` are used:

Add the variable ```p1``` to the ```run_main.sh``` or ```grid_search.sh``` script and add the variables ```--p2 $p1``` to the arguments (see the example below). Also make sure to add the variable ```p2``` to the config class in ```config.py```.


**Example:**

1. In ```run_main.sh```or ```grid_search.sh``` in the variable section set your variable
```p1='some_value'```

and add it to the arguments like this:
```
# PASS VARIABLES TO CONFIG HERE

python3 main.py --p2 $p1 --export_dir $export_dir\ # other variables...
             $dataset
```
Note that ```p2``` is the name of the variable in the config class in ```config.py``` .

2. Then in the config class add the variable ```p2``` and set it to a default value, for example ```p2 = None```

Call ```eval(Config.p2)``` when you want to use non-string variables.


Now that you know how to add and hide variables, you can also create a custom run script for your model. This can be useful if your models need different hyperparameters or settings.

## Adding Custom Loss <a name="add_components"></a> Functions, Activation Functions, Initializers and Optimizers 
In ```util/helpers.py```define an own activation function, optimizer, initialization method or loss function.
As a reference see ```activation_function``` in ```helpers.py``` and its utilisation in ```models/base_model.py``` in the method ```createModel()```:

As an example of creating a custom loss function called ```identity``` 
we define a class for the activation function and the activation function itself in ```helpers.py```:

```
class activation_function:
    @staticmethod
    def create_activation_function():
        ent_func = None
        if hasattr(activation_function, Config.ent_func):
            ent_func = getattr(activation_function, Config.ent_func)
        return ent_func

    def identity(input):
        return input
```

If you add a loss function, initializer or optimizer, create a separate class in ```helpers.py``` as done with the class ```activation_function```.


## Adding your own Model
Implement your own model by inheriting from BaseModel in ```base_model.py```.
* For training the model implement the function:

  ```def forward_emb(self, e1, r, e2): pass``` that returns a score for a subject e1, a relation r and an object e2. 

* For Entity-ranking evaluation: implement two functions:
    1. ```def scores_e2(self, E1): pass``` that returns the scores for all swapped objects for a given batch of subjects E1.

    2. ```def scores_e1(self, E2): pass``` that returns the scores for all swapped subjects for a given batch of objects E2.


* For Entity-Pair ranking evaluation implement the function:
```def score_matrix_r(self, r): pass``` that returns the scores of all swapped subjects and objects for a given relation r.

if your model needs special parameters, get them from the config class by creating an API like this:

```
class yourModel(BaseModel):

    def __init__(self, num_entities, num_relations, entity_emb_size, ent_func):
        super(yourModel, self).__init__(num_entities, num_relations, entity_emb_size, ent_func)   
        # special settings
        special_variable = None
        self.fromConfig()

    def fromConfig(self):
        self.special_variable = eval(Config.some_value)


```


## Adding your own Evaluation Protocol
Create a new Python file in the ```evaluation``` package. Access the hyperparameters and settings from the config class in ```config.py``` with the provided API: define a ```fromConfig``` method and get the information from the config class you need.
Important: Do not touch the config file inside your implementation!


Note: 
If you want to log different information than Entity-(Pair) ranking then add a separate logging method to ```util/logger```. 


## Built With

* [PyTorch](https://pytorch.org/)
* [Numpy](http://www.numpy.org/)

## Acknowledgements
Thanks to the Chair of Practical Computer Science I, University of Mannheim

