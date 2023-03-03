# AI meets Museum

This project is a joint effort of the [university of Augsburg](https://www.uni-augsburg.de/de/), [NoSQL Geeks](https://www.nosqlgeeks.com/de/index.html), and the [local museum in Krumbach](https://www.museum-krumbach.de).


* **Subject**: AI-powered tool for simplified information gathering around artifacts in a museum
* **Description**: Museums of any size are challenged to gather details about artifacts by having a limited number of human resources available. Details such as the class of an artifact, the material, the color, the age, the provenance are part of such an artifact description. The goal of this project is to create an Open Source Software tool that automatically detects properties, such as the shape, the color and the material.
* **Implementation idea**: We will evaluate several Machine Learning models in order to create vector embeddings. Finally Redis' Vector Similarity Search feature will be used to identify similarities between already classified and new artifacts.

## Prerequisite

Preparation material will be made available here:

* [Preparation](./Preparation.md)

You will also need a Redis Stack database for development purposes. You can either use a local Redis Stack database, or one of the Cloud databases that were prepared for this project. The database endpoints can be found here:

* [Databases](https://github.com/nosqlgeek/ai-meets-museum-priv)

# Project Management

We will use Github tasks and a Kanban-like board to assign tasks and keep an overview of the work that needs to be done. You can find the associated Github project here:

* [Project](https://github.com/users/nosqlgeek/projects/1)


