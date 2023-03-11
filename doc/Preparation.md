<!-- TOC -->
* [Preparation material](#preparation-material)
  * [Data cleansing and preparation](#data-cleansing-and-preparation)
  * [M(achine) L(earning) models and vector embeddings](#m--achine--l--earning--models-and-vector-embeddings)
    * [Vector embeddings](#vector-embeddings)
    * [Model selection and evaluation](#model-selection-and-evaluation)
      * [Pre-trained ML models](#pre-trained-ml-models)
      * [Model evaluation](#model-evaluation)
    * [Model training](#model-training)
    * [V(ector) S(imilarity) S(earch)](#v--ector--s--imilarity--s--earch-)
      * [Indexing method](#indexing-method)
      * [Distance functions](#distance-functions)
      * [Additional materials](#additional-materials)
  * [Redis Stack](#redis-stack)
  * [Web development](#web-development)
<!-- TOC -->


# Preparation material

This document gives the participating students some insight into the technology that must be mastered to implement our AI-powered artifact classification tool.

## Data cleansing and preparation

The source/test data is currently stored in a relational database. We will need to perform some data cleansing before this data can be used for training purposes. One idea could be to use Pandas dataframes to operate on exported CSV files. We could then use Jupyter notebooks to visualize the results of the data analysis.

A Pandas dataframe wraps data via a table structure. The Pandas documentation describes it as a `Two-dimensional, size-mutable, potentially heterogeneous tabular data.`. Pandas then allows querying and processing of the data. The query expressions are not precisely SQL, but you can also do projections and selections. So it will also be good to have a basic understanding of SQL.

NumPy is a complex mathematical library for Python. You can use it to perform calculations, e.g., statistical ones, on your data. The NumPy documentation says:

```
NumPy is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.
```

Jupyter is a software that allows us to write and manage notebooks. Such notebooks have Python code mixed in. This will enable us to do a more interactive analysis and share the results more easily with others.

Libraries like Pandas, NumPy, Matplotlib, and Seaborn are often used in Data Science projects. As said, we can use them to analyze and clean our data. Here are some example use cases:

* Render a heatmap that gives an indication of missing data
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

### Vector embeddings

ML uses, for instance, A(rtificial) N(eural) N(etwork)s to approximate a function that is hard to describe explicitly. In ANN-s, mathematically modeled neurons are used. Those neurons have weighted connections to other neurons. Training is all about finding a suitable weight matrix. We call such a trained ANN an ML model. There are a bunch of different learning algorithms that can be used to train such a model. The output of a model is typically a vector. We can embed such an output vector within a vector space to find out if any other embedded vectors are close by. Here are the steps that are involved:

1. Train an ML model `M`: Pre-trained models are already available. It's common to take a pre-trained model as the basis by doing some retraining. It's indeed also possible to train a model from scratch.
2. Embed vectors into the vector space `V`: The idea is to take our artifact data and push it through the model `M`. We embed the resulting vector into the vector space `V` for each input.  
3. Perform a similarity search `S(D)`: We use the same model `M` for a new input to retrieve the output vector. We then embed the vector into the vector space `V` and perform a search `S` for vectors that are within a specific distance `D`. 
4. Backwards mapping: An embedded vector has an identifier. This id can be mapped backwards to derive details of the object that led to this vector. This allows us to recommend or show similar objects.

You can find additional material here:

* [Blog article by Google about vector embeddings](https://cloud.google.com/blog/topics/developers-practitioners/meet-ais-multitool-vector-embeddings?hl=en)
* [Blog article about vector similarity search in Retail](https://redis.com/blog/redismart-retail-application-with-redis/)
* [Google's ML crash course](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture)

### Model selection and evaluation

#### Pre-trained ML models

We will try several models out before we decide on one that will be used in our application. As said, there are already a bunch of pre-trained models that might be a good fit for our image-based vector embeddings. 

A good class of models for image embeddings are in general:

* C(onvolutional) N(eural) N(etwork)s

Interestingly, a good model might not need to originate from the same domain. So a model trained on a 'fashion' dataset might return good vector embeddings for other product data, too. This is because the model doesn't know anything about the topic 'fashion'. It was trained to recognize shapes.

Here are some example models:

* [Yolo](https://arxiv.org/pdf/1506.02640.pdf) 
* [VGG](https://arxiv.org/pdf/1409.1556.pdf)
* [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
* [EfficientNet](https://arxiv.org/pdf/1905.11946.pdf)
* [MobileNet v3](https://arxiv.org/pdf/1905.02244.pdf)

PyTorch offers pre-trained weights for several models. The weights that are set also depend on the dataset used for training purposes. One example dataset is 'ImageNet'. Further details about pre-trained PyTorch models can be found here:

* [PyTorch - Models and Pre-Trained Weights]

In addition, here are some interesting reads:

* [Blog by Activeloop.ai](https://www.activeloop.ai/resources/generate-image-embeddings-using-a-pre-trained-cnn-and-store-them-in-hub/)
* [Blog by Romain Beaumont](https://rom1504.medium.com/image-embeddings-ed1b194d113e)

#### Model evaluation

Let's assume that we have a model `M_1` and `M_2`. How do we then actually know which model performs better? One approach could be the following one:

1. Split the amount of given data into training data and test data (e.g., 80%/20%)
2. Perform the embeddings first with the training data
3. Then test with the test data by checking how often and to which degree objects were found that were labeled (categorized) correctly

The following article describes it quite well:

* [Google's ML crash course - Training and Test Sets](https://developers.google.com/machine-learning/crash-course/training-and-test-sets/splitting-data)

> There is indeed a lot of research around the efficiency of such models, which means that you can find general information about the performance and efficiency in research papers.

### Model training

Model training can be pretty complex, so I would propose the following approach for our project:

1. Select a pre-trained model
2. Do some retraining if the time allows it
3. Only train a model from scratch if we aren't happy with the results of step 1. or step 2.


### V(ector) S(imilarity) S(earch)

We already covered VSS in step three of the section 'Vector embeddings'. Some parameters influence the quality of the search, such as the indexing method and the distance function:

#### Indexing method

The indexing approach influences the time complexity of the K(-)N(earest)N(eighbour) search that we need to perform to find close-by vectors:

* **Flat index**: The search index has a flat structure. This means that you need to perform a brute-force search to find similar vectors. The search result is accurate, but at the price of a worse time complexity when searching.
* **Indexing with a H(ierarchical) N(avigable) S(mall) W(orld) graphs**: This is an approach for the approximate K-nearest neighbor search, which achieves a logarithmic time complexity.

#### Distance functions

The following distance functions between two vectors can be used to find vectors that are within a specific distance:

* Euclidean distance
* Internal product
* Cosine distance

#### Additional materials

You can find further details here:

* [Paper about navigable small world graphs](https://arxiv.org/ftp/arxiv/papers/1603/1603.09320.pdf)
* [Youtube video that explains HNSW](https://youtu.be/QvKMwLjdK-s)
* [Blog article about distance functions in ML](https://medium.com/swlh/different-types-of-distances-used-in-machine-learning-ec7087616442)

## Redis Stack

Redis stands for Re(mote) Di(ctionary) S(erver) and is an open-source, in-memory data store. Redis Stacks extends Redis with additional data models and structures. It comes, for instance, with full-text search and VSS capabilities.

Redis Stack will help us to implement the following features:

* Object storage
* Secondary indexing and querying
* Full-text indexing and searching
* Vector embeddings and similarity search

You can find additional details about Redis Stack here:

* [Redis documentation](https://redis.io)
* [Redis Stack - Search](https://redis.io/docs/stack/search/)
* [Redis Stack VSS](https://redis.io/docs/stack/search/reference/vectors/)

## Web development

We will use a web frontend for the actual application. I think that it makes sense to stick with one programming language for all aspects of the project. Since Python is quite popular in the AI/ML space, I recommend that you familiarize yourself with some of the Python web development basics. Here are some valuable resources:

* **Web-services and app framework**: The defacto-standard for Python RESTFul services and web applications is Flask.
* **Client-side framework**: Client-side (running in a browser) frameworks, such as Angular, can add a lot of complexity. The benefit is that the usage results in a more modern web app experience. The web app code is also much better structured with an MVC framework. We might want to decide either for server-side rendering (Flask), or use a lightweight framework like Vue.js or even jQuery for this project.
* **Frontend toolkit**: Bootstrap has many reusable GUI components. 

Here are some additional resources:

* [Flask documentation](https://flask.palletsprojects.com/en/2.2.x/)
* [VueJS documentation](https://vuejs.org)
* [jQuery documentation](https://jquery.com)
* [Bootstrap documentation](https://getbootstrap.com/docs/5.3/getting-started/introduction/)
