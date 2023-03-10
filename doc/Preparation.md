# Preparation material

This document gives the participating students some insight into the technology that needs to mastered in order to implement our AI-powered artifact classification tool.

You will need to master a few things in order to be able to develop such a tool.

## Data cleansing and preparation

The source/test data is currently stored in a relational database. We will need to perform some data cleansing before this data can be used for training purposes. One idea could be to use Pandas dataframes to operate on exported CSV files. We could then use Jupyter notebooks to visualize the results of the data analysis.

A Pandas dataframe wraps data via a table structure. The Pandas documentation describes it as a `Two-dimensional, size-mutable, potentially heterogeneous tabular data.`. Pandas then allows querying and processing the data. The query expressions are not exactly SQL, but you can also do projections and selections. It would be good to have a basic understanding of SQL, too.

NumPy is a complex mathematical library for Python. You can use it to perform calculations, for instance statistics, on your data. The Numpy documentation says:

```
NumPy is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.
```

Jupyter is a software that allows us to write and manage notebooks. Such notebooks can have Python code mixed in. This allows us to a more interactive analysis and share the results more easily with others.

Libraries like Pandas, NumPy, Matplotlib, Seaborn are often used in Data Science projects. As said, we can use them to analyse and clean our data. Here are some example use cases:

* Render a heatmap which gives an indication about missing data
* Use histograms to identify missing values and outliers
* Query for duplicates by grouping by identifying properties

Here are some additional resources:

* [Pandas documentation](https://pandas.pydata.org)
* [NumPy documentation](https://numpy.org)
* [Matplotlib documentation](https://matplotlib.org)
* [Seaborn documentation](https://seaborn.pydata.org)
* [Jupyter documentation](https://jupyter.org)
* [Book 'Python for Data Analysis: Data Wrangling with pandas, NumPy, and Jupyter'](https://wesmckinney.com/book/)
* [Blog article about data cleansing](https://towardsdatascience.com/data-cleaning-in-python-the-ultimate-guide-2020-c63b88bf0a0d)

## M(achine) L(earning) models and vector embeddings

## Vector embeddings

ML uses, for instance, A(rtificial) N(eural) N(etwork)s to approximate a function that is hard to describe explicitly. In ANN-s mathematically modeled neurons are used. Those neurons have weighted connections to other neurons. Training is all about finding the right weight matrix. We call such a trained ANN an ML model. There are a bunch of different learning algorithms that can be used to train such a model. The output of a model is typically a vector. We can embed such an output vector within a vector space in order to find out if there are any other embedded vectors that are close by. Here are the steps that are involved:

1. Train an ML model `M`: There are already pre-trained models available. It's not uncommon to take a pre-trained model as the basis by doing some retraining. It's indeed also possible to train a model from scratch.
2. Embed vectors into the vector space `V`: The idea is to take our artifact data and push it through the model `M`. For each input, we embed the resulting vector into the vector space `V`.  
3. Perform a similarity search `S(D)`: For a new input, we use the same model `M` to retrieve the output vector. We then embed the vector into the vector space `V` and perform a search `S` for vectors that are within a specific distance `D`. 
4. Backwards mapping: An embedded vector has an identifier that allows us to derive details of the object that led to this vector. This allows us to recommend or show similar objects.

You can find further material here:

* [Blog article by Google about vector embeddings](https://cloud.google.com/blog/topics/developers-practitioners/meet-ais-multitool-vector-embeddings?hl=en)
* [Blog article about vector similarity search in Retail](https://redis.com/blog/redismart-retail-application-with-redis/)
* [Google's ML crash course](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture)

## Model selection and evaluation

### Pre-trained ML models

We will try a bunch of models out before we decide for one that will be used in our application. As said, there are already a bunch of pre-trained models that are a good fit for image-based vector embeddings. A good class of models for image embeddings are:

* C(onvolutional) N(eural) N(etwork)s

It's interesting that a good model might not even need to originate from the same domain. So a model that was trained on a 'fashion' dataset might actually return good vector embeddings for other product data, too. This is because the model doesn't know anything about the topic 'fashion'. It was trained to recognize shapes.

Here are some example models:

* [Yolo](https://arxiv.org/pdf/1506.02640.pdf) 
* [VGG](https://arxiv.org/pdf/1409.1556.pdf)
* [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
* [EfficientNet](https://arxiv.org/pdf/1905.11946.pdf)
* [MobileNet v3](https://arxiv.org/pdf/1905.02244.pdf)

PyTorch offers pre-trained weights for several models. The weights that are set indeed also depend on the dataset that was used for training purposes. One example dataset is 'ImageNet'. Further details about pre-trained PyTorch models can be found here:

* [PyTorch - Models and Pre-Trained Weights]

In addition, here is some interesting reads:

* [Blog by Activeloop.ai](https://www.activeloop.ai/resources/generate-image-embeddings-using-a-pre-trained-cnn-and-store-them-in-hub/)
* [Blog by Romain Beaumont](https://rom1504.medium.com/image-embeddings-ed1b194d113e)

### Model evaluation

Let's assume that we have a model `M_1` and `M_2`, how do we then actually know which model performs better? One approach could be the following one:

1. Split the amount of given data into training data and test data (e.g, 80%/20%)
2. Perform the training and embedding first with the training data
3. Then test with the test data by checking how often and to which degree objects were found that were labeled (categorized) correctly

The following article describes it quite well:

* [Google's ML crash course - Training and Test Sets](https://developers.google.com/machine-learning/crash-course/training-and-test-sets/splitting-data)

> There is indeed a lot of research around the efficiency of such models, which means that general information about the performance and efficiency can be found in research papers or the WWW.

## Model training

Model training can be quite complex, so would propose that we do the following within the scope of this project:

1. Select a pre-trained model
2. Do some retraining if the time allows it
3. Only train a model from scratch if we aren't happy with the results of step 1. or step 2.


## Vector similarity search

> TODO

## Redis Stack

> TODO

## Web development

We will use a web frontend for the actual application. We might want to stick with one programming language for all aspects of the project. Given that Python is quite popular for the AI/ML projects, I would recommend that you get yourself familar with some of the Python web development basics. Here are some useful resources:

* **Web-services and app framework**: The defacto-standard for Python RESTFul services and web applications is Flask.
* **Client-side framework**: Client-side (running in a browser) frameworks, such as Angular, can add a lot of complexity. The benefit is that the usage results in  a more modern web app experience. The web app code is also much better structured with an MVC framework. We might want to decide either for server-side rendering (Flask), or use a light-weight framework like Vue.js or even jQuery for this project.
* **Frontend toolkit**: Bootstrap comes with tons of reusable GUI components. 

Here are some additional resources:

* [Flask documentation](https://flask.palletsprojects.com/en/2.2.x/)
* [VueJS documentation](https://vuejs.org)
* [jQuery documentation](https://jquery.com)
* [Bootstrap documentation](https://getbootstrap.com/docs/5.3/getting-started/introduction/)