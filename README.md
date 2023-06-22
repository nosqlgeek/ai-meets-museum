
# ![Logo](https://github.com/nosqlgeek/ai-meets-museum/assets/63855548/6039b0dd-1a78-464b-92c6-746099e0e966)

This project is a joint effort of the [University of Applied Sciences Augsburg](https://www.hs-augsburg.de), [NoSQL Geeks](https://www.nosqlgeeks.com/de/index.html), and the [local museum in Krumbach (Schwaben)](https://www.museum-krumbach.de).

*  **Subject**: AI-powered tool for simplified information gathering around artifacts in a museum

*  **Description**: Museums of any size are challenged to gather details about artifacts by having a limited number of human resources available. Details such as the class of an artifact, the material, the color, the age, the provenance are part of such an artifact description. The goal of this project is to create an Open Source Software tool that automatically detects properties, such as the shape, the color or the material.

*  **Implementation idea**: We will evaluate several Machine Learning models in order to create vector embeddings. Finally Redis' Vector Similarity Search feature will be used to identify similarities between already classified and new artifacts.

## Getting Started
- [Installation Guide](https://github.com/nosqlgeek/ai-meets-museum/blob/main/src/aimm/docker-setup.md)
- [Product Overview](https://showcase.informatik.hs-augsburg.de/sose-2023/ai-meets-museum)

## Technology Stack

![technologien](https://github.com/nosqlgeek/ai-meets-museum/assets/63855548/b964f6e1-929a-4b48-9847-6ccc4370587a)


### Backend - [Code](https://github.com/nosqlgeek/ai-meets-museum/blob/3b6739b48c4cd2487f1c690d0b666c89488804d2/src/aimm/database.py)

#### ResNet50
We utilized the powerful ResNet50 model to generate image vector data from our source images. Leveraging the state-of-the-art deep learning capabilities of ResNet50, we have extracted rich feature representations, allowing us to efficiently store and manage our image data in our Redis database. By combining the strengths of ResNet50 and Redis, we have created a robust solution for handling image data, enabling seamless retrieval and integration within our projects.

#### Redis
With Redis, we have unlocked a robust and scalable solution for handling large volumes of data, ensuring optimal performance and seamless integration within our projects. Through our repository, you can explore our codebase, discover innovative implementations, and gain insights into how we harness Redis's capabilities to enhance data storage and retrieval.

#### Vector Embedding
Vector embedding refers to the process of representing objects, such as images, text, or other data, as fixed-length numerical vectors in a continuous vector space. In the context of Redis, vector embedding enables us to store and manipulate object data efficiently.

Redis, being an in-memory data structure store, is primarily designed for key-value storage. However, by leveraging vector embedding techniques, we can represent complex object data as compact numerical vectors that can be stored and retrieved from Redis with ease.

By using Redis to store vector embeddings, we gain several advantages. First, the compact nature of vector representations allows for efficient storage and retrieval, as the numerical vectors occupy less space compared to the original object data. Second, Redis provides fast and scalable operations, enabling quick access to the vector embeddings. This facilitates various operations, such as similarity search, recommendation systems, and data analytics, which can be performed efficiently using the stored vector data in Redis.

### Frontend [Code](https://github.com/nosqlgeek/ai-meets-museum/blob/3b6739b48c4cd2487f1c690d0b666c89488804d2/src/aimm/app.py)

#### Flask
We have leveraged the power of Flask to build an intuitive and user-friendly frontend implementation. Using Flask as the foundation, we have created a seamless bridge between our frontend and backend services, including the robust Redis database. With Flask, we have crafted a dynamic and responsive user interface that effortlessly connects users to the powerful functionality of our backend services.

#### Bootstrap
We employed the versatile Bootstrap framework to create a visually appealing and responsive frontend design. With Bootstrap, we have harnessed a comprehensive set of CSS and JavaScript components, allowing us to effortlessly build a modern and consistent user interface. By leveraging Bootstrap's grid system, responsive utilities, and extensive library of pre-built components, we have ensured a seamless user experience across various devices and screen sizes.

## Features

### Searching Similar Objects: - [Code](https://github.com/nosqlgeek/ai-meets-museum/blob/3b6739b48c4cd2487f1c690d0b666c89488804d2/src/aimm/database.py#L131-L148)
This feature enables users to search for similar objects within the application based on specific criteria or characteristics. By leveraging Redis and Flask, the application utilizes similarity algorithms to identify and retrieve objects that closely resemble the user's query.
    
### Upload New Objects: - [Code](https://github.com/nosqlgeek/ai-meets-museum/blob/3b6739b48c4cd2487f1c690d0b666c89488804d2/src/aimm/app.py#L68-L108)
With this feature, users can easily upload new objects to the application. Leveraging the power of Flask, users can seamlessly submit images, which can be then processed and can be stored in the Redis database later on. This capability allows users to contribute to the application's dataset, expanding its collection of objects and enhancing the overall functionality and breadth of the system.
    
### Getting Dataset Recommendations for New Objects: - [Code](https://github.com/nosqlgeek/ai-meets-museum/blob/3b6739b48c4cd2487f1c690d0b666c89488804d2/src/aimm/database.py#L102-L117)
This feature leverages the combination of Redis, Flask, and ResNet50 to provide users with dataset recommendations based on the new objects they upload. By analyzing the characteristics and attributes of the uploaded objects, the application suggests relevant datasets within its database. This recommendation system facilitates exploration and discovery, helping users identify datasets that align with their interests or complement the newly uploaded objects.
    
### Save New Object to Database: - [Code](https://github.com/nosqlgeek/ai-meets-museum/blob/3b6739b48c4cd2487f1c690d0b666c89488804d2/src/aimm/database.py#L79-L89)
Through the integration of Redis and Flask, this feature allows users to save newly uploaded objects directly to the database. When a user submits an object, the application stores the relevant information, such as the object's image vector and other associated data in a JSON format to the Redis database. This ensures that the object becomes part of the application's dataset and can be accessed, searched, and utilized by other users and features within the application.

## Prerequisites 

Preparation material will be made available here: 

* [Preparation](./doc/Preparation.md)  

Please join the following Discord server for Q&A:  

* [Discord](https://discord.gg/J2qERxHCPP)  

You will first only be able to communicate via the `#lobby` channel. We will then add you to the private channel `#ai-meets-museum` for any project-specific communication.  

You will also need the source datab and a Redis Stack database. You can either use a local Redis Stack database, or one of the Cloud databases that were prepared for this project. The database endpoints can be found here:  

* [Databases](https://github.com/nosqlgeek/ai-meets-museum-priv)  

Further details about the source images can be found here:  

* [Source images](./doc/SourceImages.md)  

## Functional Requirements  

A list of the functional requirements can be found here: 

* [Requirements](./doc/Requirements.md)  

## Tasks and Project Management  

We will use Github tasks and a Kanban-like board to keep an overview of the work that needs to be done. You can find the associated Github project here:  

* [Project](https://github.com/users/nosqlgeek/projects/1)
